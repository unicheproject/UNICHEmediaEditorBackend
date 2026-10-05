"""Same-AgentPlan retry reuse for deterministic physical execution nodes."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.agent.schemas import Plan
from app.api.v1.agent import get_plan_enqueuer
from app.capabilities.context import HandlerResult, OutputFile
from app.core.database import async_session_factory
from app.main import app
from app.models.agent import AgentPlan, AgentPlanAttempt, AgentSession
from app.models.asset import Asset
from app.models.enums import AgentPlanStatus, JobStatus, MediaType, TechnicalMetadataStatus
from app.models.job import Job
from app.services import jobs as jobs_svc

API = "/api/v1"


class FakeImageHandler:
    def __init__(self, capability_id: str, *, failures: int = 0) -> None:
        self.capability_id = capability_id
        self.failures = failures
        self.calls = 0

    async def run(self, ctx):
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise RuntimeError(f"controlled {self.capability_id} failure")
        assert ctx.work_dir is not None
        path = ctx.work_dir / f"{self.capability_id.replace('.', '-')}.png"
        path.write_bytes(f"output:{self.capability_id}".encode())
        return HandlerResult(outputs=[OutputFile(path, path.name, MediaType.image)])


async def _make_plan(
    client: AsyncClient, make_project, monkeypatch
) -> tuple[str, str, str, str]:
    project_id = await make_project("retry reuse")
    source_response = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("source.png", b"source-image", "image/png")},
    )
    assert source_response.status_code == 201, source_response.text
    source_id = source_response.json()["id"]
    session_response = await client.post(
        f"{API}/agent/sessions",
        json={"project_id": project_id, "asset_ids": [source_id]},
    )
    assert session_response.status_code == 201, session_response.text
    session_id = session_response.json()["id"]
    plan = Plan.model_validate({
        "type": "plan",
        "summary": "Resize then crop",
        "steps": [
            {
                "id": "resize",
                "capability_id": "image.resize",
                "asset": source_id,
                "params": {"width": 32, "height": 24},
            },
            {
                "id": "crop",
                "capability_id": "image.crop",
                "asset": "@resize",
                "params": {"x": 0, "y": 0, "width": 24, "height": 24},
            },
        ],
    })

    async def propose(*_args, **_kwargs):
        return plan, []

    monkeypatch.setattr("app.services.agent.llm.propose", propose)
    response = await client.post(
        f"{API}/agent/sessions/{session_id}/messages",
        json={"content": "Resize and crop"},
    )
    assert response.status_code == 200, response.text
    return project_id, source_id, session_id, response.json()["plan"]["id"]


def _eager_plan_enqueuer():
    from app.agent.executor import execute_plan

    async def enqueue(plan_id: uuid.UUID, attempt_id: uuid.UUID) -> None:
        async with async_session_factory() as session:
            await execute_plan(session, plan_id, attempt_id)

    return enqueue


async def test_failed_plan_retry_reuses_successful_physical_job_node(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    project_id, _, _, plan_id = await _make_plan(client, make_project, monkeypatch)
    resize = FakeImageHandler("image.resize")
    crop = FakeImageHandler("image.crop", failures=1)
    original_get_handler = jobs_svc.registry.get_handler

    def get_handler(capability_id: str):
        return {"image.resize": resize, "image.crop": crop}.get(
            capability_id, original_get_handler(capability_id)
        )

    monkeypatch.setattr(jobs_svc.registry, "get_handler", get_handler)
    app.dependency_overrides[get_plan_enqueuer] = _eager_plan_enqueuer

    first_approval = await client.post(f"{API}/agent/plans/{plan_id}/approve")
    assert first_approval.status_code == 200, first_approval.text
    first = (await client.get(f"{API}/agent/plans/{plan_id}")).json()
    assert first["status"] == "failed"
    first_resize_run = first["step_runs"][0]
    output_id = first_resize_run["output_asset_id"]
    original_source_job_id = first_resize_run["job_id"]
    async with async_session_factory() as session:
        temp_asset = await session.get(Asset, uuid.UUID(output_id))
        source_job = await session.get(Job, uuid.UUID(original_source_job_id))
        assert temp_asset is not None and temp_asset.is_intermediate
        assert source_job is not None and source_job.execution_identity
        assert source_job.project_id == uuid.UUID(project_id)
        original_expiry = temp_asset.temp_expires_at
        original_path = temp_asset.storage_path

    second_approval = await client.post(f"{API}/agent/plans/{plan_id}/approve")
    assert second_approval.status_code == 200, second_approval.text
    second = (await client.get(f"{API}/agent/plans/{plan_id}")).json()
    assert second["status"] == "succeeded", second.get("error")
    assert resize.calls == 1
    assert crop.calls == 2

    reused_run = second["step_runs"][0]
    assert reused_run["reused"] is True
    assert reused_run["job_id"] is None
    assert reused_run["source_job_id"] == original_source_job_id
    assert reused_run["source_attempt_id"] == first["attempts"][0]["id"]
    assert reused_run["output_asset_id"] == output_id
    assert second["attempts"][0]["step_runs"][0]["reused"] is False
    assert second["attempts"][1]["step_runs"][0]["reused"] is True
    assert second["attempts"][1]["job_ids"] == [second["step_runs"][1]["job_id"]]

    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(output_id))
        assert asset is not None and asset.storage_path == original_path
        assert asset.temp_expires_at == original_expiry
        total_assets = len(list(await session.scalars(
            select(Asset).where(Asset.project_id == uuid.UUID(project_id))
        )))
        all_jobs = list(await session.scalars(
            select(Job).where(Job.project_id == uuid.UUID(project_id))
        ))
        attempts = list(await session.scalars(
            select(AgentPlanAttempt).where(
                AgentPlanAttempt.plan_id == uuid.UUID(plan_id)
            ).order_by(AgentPlanAttempt.attempt_number)
        ))
        assert total_assets == 3  # original, one resize checkpoint, final crop
        assert len(all_jobs) == 3  # two first-attempt Jobs, then retry crop only
        assert attempts[0].job_ids == [
            first["step_runs"][0]["job_id"], first["step_runs"][1]["job_id"]
        ]
        assert attempts[1].job_ids == [second["step_runs"][1]["job_id"]]


async def test_active_retry_step_run_protects_reused_temp_asset_from_cleanup(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    from app.services.storage import get_storage
    from app.services.temp_assets import cleanup_expired_temp_assets

    project_id, _, _, plan_id = await _make_plan(client, make_project, monkeypatch)
    resize = FakeImageHandler("image.resize")
    crop = FakeImageHandler("image.crop", failures=1)
    original_get_handler = jobs_svc.registry.get_handler

    def get_handler(capability_id: str):
        return {"image.resize": resize, "image.crop": crop}.get(
            capability_id, original_get_handler(capability_id)
        )

    monkeypatch.setattr(jobs_svc.registry, "get_handler", get_handler)
    app.dependency_overrides[get_plan_enqueuer] = _eager_plan_enqueuer
    await client.post(f"{API}/agent/plans/{plan_id}/approve")
    first = (await client.get(f"{API}/agent/plans/{plan_id}")).json()
    temp_id = uuid.UUID(first["step_runs"][0]["output_asset_id"])

    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingCrop(FakeImageHandler):
        async def run(self, ctx):
            self.calls += 1
            entered.set()
            await release.wait()
            assert ctx.work_dir is not None
            path = ctx.work_dir / "crop.png"
            path.write_bytes(b"final-image")
            return HandlerResult(outputs=[OutputFile(path, path.name, MediaType.image)])

    blocking_crop = BlockingCrop("image.crop")
    monkeypatch.setattr(
        jobs_svc.registry,
        "get_handler",
        lambda capability_id: (
            resize if capability_id == "image.resize" else
            blocking_crop if capability_id == "image.crop" else
            original_get_handler(capability_id)
        ),
    )
    app.dependency_overrides[get_plan_enqueuer] = lambda: _background_plan_enqueuer

    async def _background_plan_enqueuer(plan_uuid: uuid.UUID, attempt_uuid: uuid.UUID):
        task = asyncio.create_task(_eager_plan_enqueuer()(plan_uuid, attempt_uuid))
        active_tasks.append(task)

    active_tasks: list[asyncio.Task] = []
    retry_response = await client.post(f"{API}/agent/plans/{plan_id}/approve")
    assert retry_response.status_code == 200, retry_response.text
    await entered.wait()

    async with async_session_factory() as session:
        asset = await session.get(Asset, temp_id)
        assert asset is not None
        original_expiry = asset.temp_expires_at
        path = get_storage().get_path(asset.storage_path)
        assert original_expiry is not None
        expiry_time = original_expiry + timedelta(seconds=1)
        assert await cleanup_expired_temp_assets(session, now=expiry_time) == 0
        assert path.is_file()
        assert asset.physical_expired_at is None

    release.set()
    await asyncio.gather(*active_tasks)
    async with async_session_factory() as session:
        asset = await session.get(Asset, temp_id)
        assert asset is not None
        assert asset.temp_expires_at == original_expiry
        assert await cleanup_expired_temp_assets(session, now=expiry_time) == 1
        assert asset.physical_expired_at == expiry_time


@pytest.mark.parametrize(
    "invalid_state", ["expired", "deadline_passed", "deleted", "missing_binary"]
)
async def test_unavailable_previous_output_is_a_cache_miss(
    client: AsyncClient, make_project, monkeypatch, invalid_state: str
) -> None:
    from app.services.storage import get_storage

    project_id, _, _, plan_id = await _make_plan(client, make_project, monkeypatch)
    resize = FakeImageHandler("image.resize")
    crop = FakeImageHandler("image.crop", failures=1)
    original_get_handler = jobs_svc.registry.get_handler

    def get_handler(capability_id: str):
        return {"image.resize": resize, "image.crop": crop}.get(
            capability_id, original_get_handler(capability_id)
        )

    monkeypatch.setattr(jobs_svc.registry, "get_handler", get_handler)
    app.dependency_overrides[get_plan_enqueuer] = _eager_plan_enqueuer
    await client.post(f"{API}/agent/plans/{plan_id}/approve")
    first = (await client.get(f"{API}/agent/plans/{plan_id}")).json()
    old_asset_id = uuid.UUID(first["step_runs"][0]["output_asset_id"])
    async with async_session_factory() as session:
        old_asset = await session.get(Asset, old_asset_id)
        assert old_asset is not None
        if invalid_state == "expired":
            old_asset.physical_expired_at = datetime.now(UTC)
            await session.commit()
        elif invalid_state == "deadline_passed":
            old_asset.temp_expires_at = datetime.now(UTC) - timedelta(seconds=1)
            await session.commit()
        elif invalid_state == "deleted":
            old_asset.deleted_at = datetime.now(UTC)
            await session.commit()
        else:
            get_storage().delete_asset(old_asset.project_id, old_asset.id)

    await client.post(f"{API}/agent/plans/{plan_id}/approve")
    second = (await client.get(f"{API}/agent/plans/{plan_id}")).json()
    assert second["status"] == "succeeded", second.get("error")
    assert resize.calls == 2
    assert second["step_runs"][0]["reused"] is False
    assert second["step_runs"][0]["output_asset_id"] != str(old_asset_id)
    async with async_session_factory() as session:
        assets = list(await session.scalars(select(Asset).where(
            Asset.project_id == uuid.UUID(project_id)
        )))
        assert len(assets) == 4  # original, old checkpoint, new checkpoint, final


async def test_reuse_policy_excludes_provider_and_multi_output_nodes() -> None:
    from app.agent.reuse import REUSE_VERSIONS

    assert "image.resize" in REUSE_VERSIONS
    assert "audio.mix" in REUSE_VERSIONS
    assert "video.split" not in REUSE_VERSIONS
    assert "video.shot.detect" not in REUSE_VERSIONS
    assert "image.caption" not in REUSE_VERSIONS
    assert "audio.transcribe" not in REUSE_VERSIONS
    assert "audio.tts" not in REUSE_VERSIONS
    assert "media.titlecard" not in REUSE_VERSIONS


async def test_fused_render_group_reuses_as_one_physical_node(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    project_id = await make_project("render-group retry reuse")
    source_response = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("source.mp4", b"source-video", "video/mp4")},
    )
    assert source_response.status_code == 201, source_response.text
    source_id = source_response.json()["id"]
    async with async_session_factory() as session:
        source = await session.get(Asset, uuid.UUID(source_id))
        assert source is not None
        source.technical_metadata_status = TechnicalMetadataStatus.available.value
        source.width, source.height, source.has_audio_stream = 160, 120, False
        await session.commit()

    session_response = await client.post(
        f"{API}/agent/sessions",
        json={"project_id": project_id, "asset_ids": [source_id]},
    )
    session_id = session_response.json()["id"]
    plan = Plan.model_validate({
        "type": "plan",
        "summary": "Resize and normalize",
        "steps": [
            {"id": "resize", "capability_id": "video.resize", "asset": source_id,
             "params": {"width": 80, "height": 60}},
            {"id": "normalize", "capability_id": "audio.normalize", "asset": "@resize",
             "params": {}},
        ],
    })

    async def propose(*_args, **_kwargs):
        return plan, []

    render_calls = 0

    async def copy_render(source_path: Path, output_path: Path, *_args, **_kwargs) -> bool:
        nonlocal render_calls
        render_calls += 1
        output_path.write_bytes(Path(source_path).read_bytes() + b"-rendered")
        return False

    normalize = FakeImageHandler("audio.normalize")
    normalize.failures = 1
    original_get_handler = jobs_svc.registry.get_handler
    monkeypatch.setattr("app.services.agent.llm.propose", propose)
    monkeypatch.setattr("app.agent.render.render_group", copy_render)
    monkeypatch.setattr(
        jobs_svc.registry,
        "get_handler",
        lambda capability_id: (
            normalize if capability_id == "audio.normalize"
            else original_get_handler(capability_id)
        ),
    )
    app.dependency_overrides[get_plan_enqueuer] = _eager_plan_enqueuer
    proposed = await client.post(
        f"{API}/agent/sessions/{session_id}/messages", json={"content": "Edit"}
    )
    plan_id = proposed.json()["plan"]["id"]
    await client.post(f"{API}/agent/plans/{plan_id}/approve")
    first = (await client.get(f"{API}/agent/plans/{plan_id}")).json()
    assert first["status"] == "failed"
    checkpoint_id = first["execution_summary"]["nodes"][0]["output_asset_id"]
    source_job_id = first["execution_summary"]["nodes"][0]["job_id"]
    async with async_session_factory() as session:
        checkpoint = await session.get(Asset, uuid.UUID(checkpoint_id))
        source_job = await session.get(Job, uuid.UUID(source_job_id))
        assert checkpoint is not None and checkpoint.is_intermediate
        assert source_job is not None and source_job.capability_id == "agent.render_group"
        assert source_job.project_id == uuid.UUID(project_id)
        original_expiry = checkpoint.temp_expires_at

    await client.post(f"{API}/agent/plans/{plan_id}/approve")
    second = (await client.get(f"{API}/agent/plans/{plan_id}")).json()
    assert second["status"] == "succeeded", second.get("error")
    assert render_calls == 1
    assert normalize.calls == 2
    group = second["execution_summary"]["nodes"][0]
    assert group["reused"] is True
    assert group["job_id"] is None
    assert group["source_job_id"] == source_job_id
    assert group["output_asset_id"] == checkpoint_id
    group_runs = [run for run in second["step_runs"] if run.get("render_group_id")]
    assert len(group_runs) == 1
    assert group_runs[0]["job_id"] is None and group_runs[0]["reused"] is True
    assert group_runs[0]["output_asset_id"] == checkpoint_id
    assert second["attempts"][1]["job_ids"] == [second["step_runs"][1]["job_id"]]
    final = await client.get(f"{API}/assets/{second['result_asset_ids'][0]}")
    assert final.status_code == 200
    assert final.json()["is_intermediate"] is False
    async with async_session_factory() as session:
        checkpoint = await session.get(Asset, uuid.UUID(checkpoint_id))
        assert checkpoint is not None and checkpoint.temp_expires_at == original_expiry
        group_jobs = list(await session.scalars(select(Job).where(
            Job.project_id == uuid.UUID(project_id),
            Job.capability_id == "agent.render_group",
        )))
        assert len(group_jobs) == 1


async def test_reuse_identity_canonicalizes_defaults_and_binds_all_inputs(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    from app.agent import reuse

    project_id = uuid.UUID(await make_project("identity inputs"))
    video = (await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("source.mp4", b"video-content", "video/mp4")},
    )).json()
    audio_one = (await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("music-a.wav", b"audio-a", "audio/wav")},
    )).json()
    audio_two = (await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("music-b.wav", b"audio-b", "audio/wav")},
    )).json()
    async with async_session_factory() as session:
        base = await reuse.build_execution_identity(
            session,
            project_id=project_id,
            capability_id="audio.mix",
            asset_id=uuid.UUID(video["id"]),
            params={"music_asset_id": audio_one["id"]},
        )
        explicit_defaults = await reuse.build_execution_identity(
            session,
            project_id=project_id,
            capability_id="audio.mix",
            asset_id=uuid.UUID(video["id"]),
            params={
                "music_asset_id": audio_one["id"],
                "music_volume": 0.3,
                "mode": "mix",
            },
        )
        other_aux = await reuse.build_execution_identity(
            session,
            project_id=project_id,
            capability_id="audio.mix",
            asset_id=uuid.UUID(video["id"]),
            params={"music_asset_id": audio_two["id"]},
        )
        assert base == explicit_defaults
        assert base != other_aux

        source = await session.get(Asset, uuid.UUID(video["id"]))
        assert source is not None
        source.checksum_sha256 = "f" * 64
        await session.commit()
        changed_source = await reuse.build_execution_identity(
            session,
            project_id=project_id,
            capability_id="audio.mix",
            asset_id=uuid.UUID(video["id"]),
            params={"music_asset_id": audio_one["id"]},
        )
        assert changed_source != base

        changed_parameters = await reuse.build_execution_identity(
            session,
            project_id=project_id,
            capability_id="audio.mix",
            asset_id=uuid.UUID(video["id"]),
            params={"music_asset_id": audio_one["id"], "music_volume": 0.6},
        )
        assert changed_parameters != changed_source

        current_version = reuse.REUSE_VERSIONS["audio.mix"]
        monkeypatch.setitem(reuse.REUSE_VERSIONS, "audio.mix", "next-version")
        changed_version = await reuse.build_execution_identity(
            session,
            project_id=project_id,
            capability_id="audio.mix",
            asset_id=uuid.UUID(video["id"]),
            params={"music_asset_id": audio_one["id"]},
        )
        assert changed_version != changed_source
        assert current_version != reuse.REUSE_VERSIONS["audio.mix"]


@pytest.mark.parametrize(
    "source_status", [JobStatus.failed, JobStatus.cancelled, JobStatus.running]
)
async def test_non_successful_physical_node_is_never_reused(
    client: AsyncClient, make_project, source_status: JobStatus
) -> None:
    from app.agent.reuse import find_reusable_output

    project_id = uuid.UUID(await make_project("non-success cache source"))
    upload = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("output.png", b"complete-output", "image/png")},
    )
    output_id = uuid.UUID(upload.json()["id"])
    async with async_session_factory() as session:
        output_asset = await session.get(Asset, output_id)
        assert output_asset is not None
        output_asset.technical_metadata_status = TechnicalMetadataStatus.available.value
        agent_session = AgentSession(project_id=project_id, asset_ids=[])
        session.add(agent_session)
        await session.flush()
        plan = AgentPlan(
            session_id=agent_session.id,
            status=AgentPlanStatus.failed,
            steps=[], step_runs=[], result_asset_ids=[],
        )
        session.add(plan)
        await session.flush()
        source_attempt = AgentPlanAttempt(
            plan_id=plan.id, attempt_number=1, status="failed", job_ids=[],
            result_asset_ids=[], step_runs=[],
        )
        current_attempt = AgentPlanAttempt(
            plan_id=plan.id, attempt_number=2, status="running", job_ids=[],
            result_asset_ids=[], step_runs=[],
        )
        session.add_all([source_attempt, current_attempt])
        await session.flush()
        session.add(Job(
            project_id=project_id,
            capability_id="image.resize",
            status=source_status,
            input={},
            output={"outputs": [{
                "asset_id": str(output_id), "media_type": "image", "filename": "out.png",
            }]},
            finished_at=None if source_status == JobStatus.running else datetime.now(UTC),
            agent_plan_id=plan.id,
            agent_attempt_id=source_attempt.id,
            execution_identity="a" * 64,
        ))
        await session.commit()
        assert await find_reusable_output(
            session,
            plan_id=plan.id,
            current_attempt=current_attempt,
            project_id=project_id,
            identity="a" * 64,
        ) is None


async def test_lookup_never_crosses_agent_plan_or_project(
    client: AsyncClient, make_project
) -> None:
    from app.agent.reuse import find_reusable_output

    first_project = uuid.UUID(await make_project("reuse scope A"))
    second_project = uuid.UUID(await make_project("reuse scope B"))
    upload = await client.post(
        f"{API}/projects/{first_project}/assets",
        files={"file": ("output.png", b"complete-output", "image/png")},
    )
    output_id = uuid.UUID(upload.json()["id"])
    async with async_session_factory() as session:
        asset = await session.get(Asset, output_id)
        assert asset is not None
        asset.technical_metadata_status = TechnicalMetadataStatus.available.value
        source_session = AgentSession(project_id=first_project, asset_ids=[])
        target_session = AgentSession(project_id=first_project, asset_ids=[])
        session.add_all([source_session, target_session])
        await session.flush()
        source_plan = AgentPlan(
            session_id=source_session.id, status=AgentPlanStatus.failed,
            steps=[], step_runs=[], result_asset_ids=[],
        )
        target_plan = AgentPlan(
            session_id=target_session.id, status=AgentPlanStatus.failed,
            steps=[], step_runs=[], result_asset_ids=[],
        )
        session.add_all([source_plan, target_plan])
        await session.flush()
        source_attempt = AgentPlanAttempt(
            plan_id=source_plan.id, attempt_number=1, status="failed", job_ids=[],
            result_asset_ids=[], step_runs=[],
        )
        target_attempt = AgentPlanAttempt(
            plan_id=target_plan.id, attempt_number=2, status="running", job_ids=[],
            result_asset_ids=[], step_runs=[],
        )
        session.add_all([source_attempt, target_attempt])
        await session.flush()
        session.add(Job(
            project_id=first_project,
            capability_id="image.resize",
            status=JobStatus.succeeded,
            input={},
            output={"outputs": [{
                "asset_id": str(output_id), "media_type": "image", "filename": "out.png",
            }]},
            progress=100,
            finished_at=datetime.now(UTC),
            agent_plan_id=source_plan.id,
            agent_attempt_id=source_attempt.id,
            execution_identity="b" * 64,
        ))
        await session.commit()
        assert await find_reusable_output(
            session,
            plan_id=target_plan.id,
            current_attempt=target_attempt,
            project_id=first_project,
            identity="b" * 64,
        ) is None
        assert await find_reusable_output(
            session,
            plan_id=source_plan.id,
            current_attempt=source_attempt,
            project_id=second_project,
            identity="b" * 64,
        ) is None


def test_render_group_identity_includes_operation_order_and_compiler_version() -> None:
    from app.agent import render, reuse

    asset_id = "22222222-2222-2222-2222-222222222222"
    facts = {asset_id: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "a" * 64,
        "technical_metadata_status": "available", "width": 160, "height": 120,
        "has_audio_stream": False,
    }}
    crop_then_resize = Plan.model_validate({"type": "plan", "steps": [
        {"id": "crop", "capability_id": "video.crop", "asset": asset_id,
         "params": {"x": 0, "y": 0, "width": 120, "height": 100}},
        {"id": "resize", "capability_id": "video.resize", "asset": "@crop",
         "params": {"width": 80, "height": 60}},
    ]})
    resize_then_crop = Plan.model_validate({"type": "plan", "steps": [
        {"id": "resize", "capability_id": "video.resize", "asset": asset_id,
         "params": {"width": 80, "height": 60}},
        {"id": "crop", "capability_id": "video.crop", "asset": "@resize",
         "params": {"x": 0, "y": 0, "width": 40, "height": 30}},
    ]})
    first = render.compile_execution_plan(crop_then_resize, facts)["nodes"][0]
    second = render.compile_execution_plan(resize_then_crop, facts)["nodes"][0]
    key_one = reuse.render_group_identity(
        uuid.UUID("11111111-1111-1111-1111-111111111111"),
        first,
        optimizer_version=render.OPTIMIZER_VERSION,
    )
    key_two = reuse.render_group_identity(
        uuid.UUID("11111111-1111-1111-1111-111111111111"),
        second,
        optimizer_version=render.OPTIMIZER_VERSION,
    )
    key_new_version = reuse.render_group_identity(
        uuid.UUID("11111111-1111-1111-1111-111111111111"),
        first,
        optimizer_version="future-compiler",
    )
    assert key_one != key_two
    assert key_one != key_new_version
