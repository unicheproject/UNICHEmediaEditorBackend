"""Durable one-attempt claims and cancellation-safe output publication."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.dialects import postgresql, sqlite

from app.agent.executor import execute_plan
from app.agent.schemas import Plan
from app.capabilities import registry
from app.capabilities.context import HandlerResult, JobContext, OutputFile
from app.core.config import settings
from app.core.database import async_session_factory
from app.core.errors import ValidationError
from app.models.asset import Asset
from app.models.enums import AgentPlanStatus, JobStatus, MediaType
from app.models.job import Job
from app.schemas.job import JobCreate
from app.services import agent as agent_svc
from app.services import assets as assets_svc
from app.services import jobs as jobs_svc
from app.services.storage import get_storage
from app.workers.tasks import run_job


class CountingHandler:
    def __init__(self, outputs: int = 1) -> None:
        self.calls = 0
        self.outputs = outputs
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.fail = False

    async def run(self, ctx: JobContext) -> HandlerResult:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        if self.fail:
            raise RuntimeError("handler failed")
        files = []
        for index in range(self.outputs):
            name = f"result-{index}.srt"
            path = ctx.out_path(name)
            path.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n")
            files.append(OutputFile(path, name, MediaType.subtitle))
        return HandlerResult(data={"done": True}, outputs=files)


async def make_job(make_project, monkeypatch, handler: CountingHandler) -> uuid.UUID:
    project_id = uuid.UUID(await make_project("Fencing"))
    monkeypatch.setitem(registry._HANDLERS, "media.titlecard", handler)
    async with async_session_factory() as session:
        job = await jobs_svc.create_job(session, JobCreate(
            capability_id="media.titlecard", project_id=project_id,
            input={"text": "Test"},
        ))
        return job.id


async def snapshot(job_id: uuid.UUID) -> tuple[Job, list[Asset]]:
    async with async_session_factory() as session:
        job = await jobs_svc.get_job(session, job_id)
        assets = list((await session.scalars(select(Asset))).all())
        return job, assets


def stored_files() -> list[Path]:
    return list(Path(settings.storage_dir).glob("projects/*/assets/*/original/*"))


async def test_sequential_delivery_publishes_once(make_project, monkeypatch) -> None:
    handler = CountingHandler()
    job_id = await make_job(make_project, monkeypatch, handler)
    before = set(stored_files())
    assert await run_job({}, str(job_id)) == "succeeded"
    first, assets = await snapshot(job_id)
    output = first.output
    assert len(assets) == 1
    assert await run_job({}, str(job_id)) == "succeeded"
    second, assets = await snapshot(job_id)
    assert handler.calls == 1
    assert len(assets) == 1
    assert second.output == output
    assert len(set(stored_files()) - before) == 1
    assert output["outputs"][0]["asset_id"] == str(assets[0].id)
    assert (assets[0].provenance_kind, assets[0].ai_provenance, assets[0].source_job_id) == (
        "derived", "none", job_id,
    )


@pytest.mark.parametrize("status", [
    JobStatus.running, JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled,
])
async def test_nonqueued_delivery_is_noop(make_project, monkeypatch, status) -> None:
    handler = CountingHandler()
    job_id = await make_job(make_project, monkeypatch, handler)
    async with async_session_factory() as session:
        job = await jobs_svc.get_job(session, job_id)
        job.status = status
        job.output = {"existing": True}
        job.error = "existing error"
        await session.commit()
    before = set(stored_files())
    assert await run_job({}, str(job_id)) == status.value
    job, assets = await snapshot(job_id)
    assert job.status == status and job.output == {"existing": True}
    assert job.error == "existing error"
    assert handler.calls == 0 and assets == []
    assert set(stored_files()) == before


async def test_concurrent_independent_sessions_have_one_claim(make_project, monkeypatch) -> None:
    handler = CountingHandler()
    job_id = await make_job(make_project, monkeypatch, handler)
    barrier = asyncio.Event()
    arrivals = 0
    claim = jobs_svc.claim_job

    async def synchronized_claim(session, identity):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            barrier.set()
        await asyncio.wait_for(barrier.wait(), 5)
        return await claim(session, identity)

    monkeypatch.setattr(jobs_svc, "claim_job", synchronized_claim)
    results = await asyncio.gather(run_job({}, str(job_id)), run_job({}, str(job_id)))
    assert set(results) <= {"running", "succeeded"}
    job, assets = await snapshot(job_id)
    assert job.status == JobStatus.succeeded
    assert handler.calls == 1 and len(assets) == 1


async def test_claim_uses_conditional_sql_and_handler_has_no_transaction(
    make_project, monkeypatch,
) -> None:
    handler = CountingHandler()
    job_id = await make_job(make_project, monkeypatch, handler)
    async with async_session_factory() as session:
        execute = session.execute
        statements = []

        async def capture(statement, *args, **kwargs):
            statements.append(statement)
            return await execute(statement, *args, **kwargs)

        run = handler.run

        async def check_transaction(ctx):
            assert not session.in_transaction()
            return await run(ctx)

        monkeypatch.setattr(session, "execute", capture)
        monkeypatch.setattr(handler, "run", check_transaction)
        await jobs_svc.execute_job(session, job_id)
    for dialect in (sqlite.dialect(), postgresql.dialect()):
        sql = str(statements[0].compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
        assert "UPDATE jobs" in sql and "jobs.status = 'queued'" in sql
        publication = next(statement for statement in statements
                           if getattr(statement, "is_update", False)
                           and statement.compile().params.get("status") == JobStatus.succeeded)
        sql = str(publication.compile(dialect=dialect, compile_kwargs={"literal_binds": False}))
        assert "jobs.status =" in sql and "output=" in sql


async def test_queued_cancellation_prevents_execution(make_project, monkeypatch) -> None:
    handler = CountingHandler()
    job_id = await make_job(make_project, monkeypatch, handler)
    async with async_session_factory() as session:
        await jobs_svc.cancel_job(session, job_id)
    assert await run_job({}, str(job_id)) == "cancelled"
    job, assets = await snapshot(job_id)
    assert handler.calls == 0 and not assets and job.output is None


@pytest.mark.parametrize("handler_fails", [False, True])
async def test_running_cancellation_prevents_publication_and_stale_failure(
    make_project, monkeypatch, handler_fails,
) -> None:
    handler = CountingHandler()
    handler.release.clear()
    handler.fail = handler_fails
    job_id = await make_job(make_project, monkeypatch, handler)
    before = set(stored_files())
    task = asyncio.create_task(run_job({}, str(job_id)))
    try:
        await asyncio.wait_for(handler.entered.wait(), 5)
        async with async_session_factory() as session:
            await jobs_svc.cancel_job(session, job_id)
    finally:
        handler.release.set()
    assert await task == "cancelled"
    job, assets = await snapshot(job_id)
    assert job.error == "Cancelled by user" and job.output is None
    assert not assets and set(stored_files()) == before


async def test_cancellation_after_preparation_cleans_objects(make_project, monkeypatch) -> None:
    handler = CountingHandler(2)
    job_id = await make_job(make_project, monkeypatch, handler)
    prepared = asyncio.Event()
    release = asyncio.Event()
    prepare = assets_svc.prepare_derived_asset
    count = 0

    async def paused_prepare(**kwargs):
        nonlocal count
        asset = await prepare(**kwargs)
        count += 1
        if count == 2:
            prepared.set()
            await release.wait()
        return asset

    monkeypatch.setattr(assets_svc, "prepare_derived_asset", paused_prepare)
    before = set(stored_files())
    task = asyncio.create_task(run_job({}, str(job_id)))
    try:
        await asyncio.wait_for(prepared.wait(), 5)
        job, assets = await snapshot(job_id)
        assert job.status == JobStatus.running and not assets and job.output is None
        async with async_session_factory() as session:
            await jobs_svc.cancel_job(session, job_id)
    finally:
        release.set()
    assert await task == "cancelled"
    assert (await snapshot(job_id))[1] == []
    assert set(stored_files()) == before


async def test_stale_cancellation_cannot_overwrite_publication(make_project, monkeypatch) -> None:
    handler = CountingHandler()
    handler.release.clear()
    job_id = await make_job(make_project, monkeypatch, handler)
    task = asyncio.create_task(run_job({}, str(job_id)))
    async with async_session_factory() as stale_session:
        await asyncio.wait_for(handler.entered.wait(), 5)
        stale = await jobs_svc.get_job(stale_session, job_id)
        assert stale.status == JobStatus.running
        handler.release.set()
        assert await task == "succeeded"
        with pytest.raises(ValidationError, match="already succeeded"):
            await jobs_svc.cancel_job(stale_session, job_id)
        assert stale.status == JobStatus.succeeded
    job, assets = await snapshot(job_id)
    assert len(assets) == 1 and job.output and job.error is None


async def test_multioutput_publishes_complete_set_atomically(make_project, monkeypatch) -> None:
    handler = CountingHandler(2)
    job_id = await make_job(make_project, monkeypatch, handler)
    assert await run_job({}, str(job_id)) == "succeeded"
    job, assets = await snapshot(job_id)
    assert {out["asset_id"] for out in job.output["outputs"]} == {str(asset.id) for asset in assets}
    assert len(assets) == 2


async def test_second_output_copy_failure_publishes_none(make_project, monkeypatch) -> None:
    handler = CountingHandler(2)
    job_id = await make_job(make_project, monkeypatch, handler)
    storage = get_storage()
    save = storage.save_upload
    copies = 0

    def broken_copy(*args):
        nonlocal copies
        value = save(*args)
        copies += 1
        if copies == 2:
            raise OSError("second copy failed after writing")
        return value

    monkeypatch.setattr(storage, "save_upload", broken_copy)
    monkeypatch.setattr(assets_svc, "get_storage", lambda: storage)
    before = set(stored_files())
    assert await run_job({}, str(job_id)) == "failed"
    job, assets = await snapshot(job_id)
    assert copies == 2 and not assets and job.output is None
    assert set(stored_files()) == before


async def test_publication_insert_failure_rolls_back_all_and_cleans_storage(
    make_project, monkeypatch,
) -> None:
    handler = CountingHandler(2)
    job_id = await make_job(make_project, monkeypatch, handler)
    before = set(stored_files())
    async with async_session_factory() as session:
        def break_second_asset(sync_session, *_):
            assets = [row for row in sync_session.new if isinstance(row, Asset)]
            if len(assets) == 2:
                assets[1].filename = None  # actual NOT NULL DB failure, not a fake rollback

        event.listen(session.sync_session, "before_flush", break_second_asset)
        result = await jobs_svc.execute_job(session, job_id)
        assert result.status == JobStatus.failed
        assert await session.scalar(select(func.count()).select_from(Asset)) == 0
    job, assets = await snapshot(job_id)
    assert not assets and job.output is None
    assert set(stored_files()) == before


async def test_process_loss_before_publication_has_no_authoritative_rows(
    make_project, monkeypatch,
) -> None:
    class SimulatedProcessLoss(BaseException):
        pass

    handler = CountingHandler(2)
    job_id = await make_job(make_project, monkeypatch, handler)
    async with async_session_factory() as session:
        execute = session.execute

        async def disappear(statement, *args, **kwargs):
            if getattr(statement, "is_update", False) and statement.table.name == "jobs":
                values = statement.compile().params
                if values.get("status") == JobStatus.succeeded:
                    raise SimulatedProcessLoss
            return await execute(statement, *args, **kwargs)

        monkeypatch.setattr(session, "execute", disappear)
        with pytest.raises(SimulatedProcessLoss):
            await jobs_svc.execute_job(session, job_id)
    job, assets = await snapshot(job_id)
    assert job.status == JobStatus.running and job.output is None and not assets
    assert await run_job({}, str(job_id)) == "running"
    assert handler.calls == 1


async def test_cancelled_task_does_not_resurrect_on_redelivery(make_project, monkeypatch) -> None:
    handler = CountingHandler()
    handler.release.clear()
    job_id = await make_job(make_project, monkeypatch, handler)
    task = asyncio.create_task(run_job({}, str(job_id)))
    await asyncio.wait_for(handler.entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await run_job({}, str(job_id)) == "cancelled"
    assert handler.calls == 1 and (await snapshot(job_id))[1] == []


async def test_publication_transaction_wins_before_concurrent_cancellation(
    make_project, monkeypatch,
) -> None:
    handler = CountingHandler(2)
    job_id = await make_job(make_project, monkeypatch, handler)
    fenced = asyncio.Event()
    release = asyncio.Event()
    async with async_session_factory() as publishing:
        execute = publishing.execute

        async def pause_after_fence(statement, *args, **kwargs):
            result = await execute(statement, *args, **kwargs)
            if (getattr(statement, "is_update", False)
                    and statement.compile().params.get("status") == JobStatus.succeeded):
                fenced.set()
                await release.wait()
            return result

        monkeypatch.setattr(publishing, "execute", pause_after_fence)
        task = asyncio.create_task(jobs_svc.execute_job(publishing, job_id))
        await asyncio.wait_for(fenced.wait(), 5)
        # Uncommitted success and prepared output rows are not externally visible.
        before, rows = await snapshot(job_id)
        assert before.status == JobStatus.running and rows == [] and before.output is None
        async with async_session_factory() as cancelling:
            cancel = asyncio.create_task(jobs_svc.cancel_job(cancelling, job_id))
            release.set()
            assert (await task).status == JobStatus.succeeded
            with pytest.raises(ValidationError, match="already succeeded"):
                await cancel
    job, rows = await snapshot(job_id)
    assert job.status == JobStatus.succeeded and len(rows) == 2


@pytest.mark.parametrize("lost_acknowledgement", [False, True])
async def test_commit_interruption_keeps_authoritative_outputs(
    make_project, monkeypatch, lost_acknowledgement,
) -> None:
    handler = CountingHandler()
    job_id = await make_job(make_project, monkeypatch, handler)
    committed = asyncio.Event()
    release = asyncio.Event()
    async with async_session_factory() as session:
        commit = session.commit

        async def interrupted_commit():
            publishing = any(isinstance(row, Asset) for row in session.new)
            await commit()
            if publishing:
                if lost_acknowledgement:
                    raise OSError("simulated lost commit acknowledgement")
                committed.set()
                await release.wait()

        monkeypatch.setattr(session, "commit", interrupted_commit)
        task = asyncio.create_task(jobs_svc.execute_job(session, job_id))
        if lost_acknowledgement:
            assert (await task).status == JobStatus.succeeded
        else:
            await asyncio.wait_for(committed.wait(), 5)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
    job, rows = await snapshot(job_id)
    assert job.status == JobStatus.succeeded and len(rows) == 1
    assert get_storage().get_path(rows[0].storage_path).is_file()
    assert await run_job({}, str(job_id)) == "succeeded"
    assert handler.calls == 1


async def setup_agent(client, make_project, monkeypatch, *, fused: bool):
    project_id = await make_project("Agent fencing")
    scope = []
    if fused:
        uploaded = await client.post(f"/api/v1/projects/{project_id}/assets", files={
            "file": ("source.mp4", b"source-video", "video/mp4"),
        })
        source_id = uploaded.json()["id"]
        scope = [source_id]
        async with async_session_factory() as session:
            source = await session.get(Asset, uuid.UUID(source_id))
            source.technical_metadata_status = "available"
            source.width, source.height, source.has_audio_stream = 160, 120, False
            await session.commit()
        step = {"id": "resize", "capability_id": "video.resize", "asset": source_id,
                "params": {"width": 80, "height": 60}}
    else:
        step = {"id": "title", "capability_id": "media.titlecard", "params": {"text": "Test"}}
    response = await client.post("/api/v1/agent/sessions", json={
        "project_id": project_id, "asset_ids": scope,
    })
    session_id = response.json()["id"]

    async def propose(*_args, **_kwargs):
        return Plan.model_validate({"type": "plan", "summary": "Test", "steps": [step]}), []

    monkeypatch.setattr("app.services.agent.llm.propose", propose)
    response = await client.post(f"/api/v1/agent/sessions/{session_id}/messages", json={
        "content": "Test",
    })
    assert response.status_code == 200, response.text
    plan_id = uuid.UUID(response.json()["plan"]["id"])
    async with async_session_factory() as session:
        _, attempt, _, _ = await agent_svc.approve_plan(session, plan_id)
        attempt_id = attempt.id
    return uuid.UUID(project_id), plan_id, attempt_id


async def test_agent_parent_and_standard_physical_redelivery_are_noop(
    client, make_project, monkeypatch,
) -> None:
    handler = CountingHandler()
    monkeypatch.setitem(registry._HANDLERS, "media.titlecard", handler)
    project_id, plan_id, attempt_id = await setup_agent(
        client, make_project, monkeypatch, fused=False,
    )
    async with async_session_factory() as session:
        first = await execute_plan(session, plan_id, attempt_id)
        assert first.status == AgentPlanStatus.succeeded
        physical_id = uuid.UUID(first.step_runs[0]["job_id"])
        duplicate = await execute_plan(session, plan_id, attempt_id)
        assert duplicate.status == AgentPlanStatus.succeeded
    assert await run_job({}, str(physical_id)) == "succeeded"
    async with async_session_factory() as session:
        outputs = list(await session.scalars(select(Asset).where(Asset.project_id == project_id)))
        assert len(outputs) == 1
    assert handler.calls == 1


@pytest.mark.parametrize("fused", [False, True])
async def test_agent_physical_cancellation_prevents_output_and_parent_success(
    client, make_project, monkeypatch, fused,
) -> None:
    handler = CountingHandler()
    handler.release.clear()
    monkeypatch.setitem(registry._HANDLERS, "media.titlecard", handler)

    async def paused_render(source, destination, *_args, **_kwargs):
        handler.entered.set()
        await handler.release.wait()
        destination.write_bytes(Path(source).read_bytes())
        return False

    monkeypatch.setattr("app.agent.render.render_group", paused_render)
    project_id, plan_id, attempt_id = await setup_agent(
        client, make_project, monkeypatch, fused=fused,
    )

    async def execute():
        async with async_session_factory() as session:
            await execute_plan(session, plan_id, attempt_id)

    task = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(handler.entered.wait(), 5)
        async with async_session_factory() as session:
            physical = await session.scalar(select(Job).where(Job.project_id == project_id))
            physical_id = physical.id
            await jobs_svc.cancel_job(session, physical_id)
    finally:
        handler.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with async_session_factory() as session:
        plan = await agent_svc.get_plan(session, plan_id)
        assert plan.status == AgentPlanStatus.failed
        assert plan.result_asset_ids == []
        assert plan.step_runs[0]["status"] == "cancelled"
        count = await session.scalar(select(func.count()).select_from(Asset)
                                     .where(Asset.project_id == project_id))
        assert count == int(fused)  # only the uploaded render-group source survives
    assert await run_job({}, str(physical_id)) == "cancelled"


async def test_completed_render_group_job_redelivery_is_noop(client, make_project, monkeypatch):
    calls = 0

    async def render(source, destination, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        destination.write_bytes(Path(source).read_bytes())
        return False

    monkeypatch.setattr("app.agent.render.render_group", render)
    project_id, plan_id, attempt_id = await setup_agent(
        client, make_project, monkeypatch, fused=True,
    )
    async with async_session_factory() as session:
        plan = await execute_plan(session, plan_id, attempt_id)
        assert plan.status == AgentPlanStatus.succeeded
        job_id = uuid.UUID(plan.step_runs[0]["job_id"])
        output_ids = list(plan.result_asset_ids)
    assert await run_job({}, str(job_id)) == "succeeded"
    async with async_session_factory() as session:
        outputs = list(await session.scalars(select(Asset).where(Asset.project_id == project_id)))
        assert len(outputs) == 2
        job = await session.get(Job, job_id)
        assert job.output["outputs"][0]["asset_id"] == output_ids[0]
    assert calls == 1
