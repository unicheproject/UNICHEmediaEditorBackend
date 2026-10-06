"""Ordinary dispatch compensation; real DB and ARQ inspection, no Redis I/O."""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from arq.jobs import serialize_job
from arq.utils import timestamp_ms
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.dialects import postgresql, sqlite

from app.api.v1.agent import get_plan_enqueuer
from app.api.v1.jobs import get_enqueuer
from app.capabilities import registry
from app.core.database import async_session_factory
from app.core.errors import AppError
from app.main import app
from app.models.agent import AgentPlan, AgentPlanAttempt
from app.models.asset import Asset
from app.models.enums import JobStatus
from app.models.job import Job
from app.schemas.job import JobCreate
from app.services import jobs as svc
from app.services.media_metadata import probe_asset_metadata
from app.workers import queue
from app.workers.queue import DispatchOutcome
from app.workers.tasks import run_job
from tests.test_agent_lifecycle import _create_plan
from tests.test_job_fencing import CountingHandler
from tests.test_project_export import (
    NEED_MEDIA_TOOLS,
    _document,
    _make_video,
    _save,
    _upload_video,
)

API = "/api/v1"
RAW_DIAGNOSTIC = "internal transport diagnostics must not be public"
ERROR_BODY = {"error": {"code": "internal_error", "message": svc.DISPATCH_FAILURE_MESSAGE}}


class InspectionPipeline:
    """Redis wire boundary for the installed ARQ Job.status() implementation."""

    def __init__(self, pool):
        self.pool = pool
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def exists(self, key):
        self.commands.append(("exists", key))

    def zscore(self, key, identity):
        self.commands.append(("zscore", identity))

    async def execute(self):
        if self.pool.inspection_error:
            raise ConnectionError(RAW_DIAGNOSTIC)
        values = []
        for command, key in self.commands:
            if command == "zscore":
                values.append(await self.pool.zscore(self.pool.default_queue_name, key))
            elif key.startswith("arq:result:"):
                values.append(self.pool.state == "complete")
            else:
                values.append(self.pool.state == "in_progress")
        return values


class FakePool:
    default_queue_name = "arq:queue"
    job_deserializer = None

    def __init__(self, mode="accepted", state="queued"):
        self.mode = mode
        self.state = state
        self.inspection_error = False
        self.tasks = {}
        self.calls = []
        self.closed = False
        self.committed_before_enqueue = False

    async def enqueue_job(self, function, *args, **kwargs):
        identity = kwargs["_job_id"]
        self.calls.append((function, args, kwargs))
        if function == queue.RUN_JOB_TASK:
            async with async_session_factory() as session:
                persisted = await session.get(Job, uuid.UUID(identity))
                assert persisted is not None and persisted.status == JobStatus.queued
                self.committed_before_enqueue = True
        if self.mode == "raise":
            self.state = "not_found"
            raise ConnectionError(RAW_DIAGNOSTIC)
        if self.state not in {"complete", "not_found", "orphan"}:
            self.tasks[identity] = serialize_job(function, args, {}, None, timestamp_ms())
        if self.mode in {"unknown", "lost_unknown"}:
            self.inspection_error = True
        if self.mode in {"lost_ack", "lost_unknown"}:
            raise TimeoutError(RAW_DIAGNOSTIC)
        if self.mode in {"existing", "unknown"}:
            return None
        return SimpleNamespace(job_id=identity)

    def pipeline(self, **_kwargs):
        return InspectionPipeline(self)

    async def zscore(self, _name, _identity):
        if self.state in {"queued", "orphan", "in_progress"}:
            return timestamp_ms() - 1000
        if self.state == "deferred":
            return timestamp_ms() + 10000
        return None

    async def get(self, key):
        # Job.info() tries the result first, then decodes the serialized JobDef.
        return self.tasks.get(key.removeprefix("arq:job:"))

    async def aclose(self):
        self.closed = True
        if self.mode == "close_error":
            raise ConnectionError(RAW_DIAGNOSTIC)


def install_queue(monkeypatch, pool):
    async def get_pool():
        return pool

    monkeypatch.setattr(queue, "get_arq_pool", get_pool)
    app.dependency_overrides[get_enqueuer] = lambda: queue.enqueue_job


async def request_body(client, make_project, tmp_path, producer):
    pid = await make_project("Dispatch")
    if producer == "generic":
        return pid, {"capability_id": "media.titlecard", "project_id": pid,
                     "input": {"text": "Test"}}
    source = tmp_path / "source.mp4"
    _make_video(source)
    aid = await _upload_video(client, pid, source)
    async with async_session_factory() as session:
        await probe_asset_metadata(session, uuid.UUID(aid))
    if producer == "compose":
        return pid, {"capability_id": "video.compose", "project_id": pid,
                     "input": {"asset_ids": [aid], "width": 96, "height": 64}}
    revision = await _save(client, pid, _document([aid]))
    return pid, {"capability_id": "project.export", "project_id": pid,
                 "input": {"composition_revision_id": revision["revision_id"], "preset": "web"}}


async def snapshot():
    async with async_session_factory() as session:
        jobs = list(await session.scalars(select(Job)))
        derived = list(await session.scalars(select(Asset)))
        return jobs, derived


@NEED_MEDIA_TOOLS
@pytest.mark.parametrize("producer", ["generic", "compose", "export"])
@pytest.mark.parametrize("mode,state", [
    ("raise", "not_found"), ("existing", "complete"),
    ("existing", "not_found"), ("existing", "orphan"),
    ("unknown", "not_found"), ("lost_unknown", "queued"),
])
async def test_rejected_or_unconfirmable_dispatch_is_failed(
    client, make_project, tmp_path, monkeypatch, producer, mode, state,
):
    pid, body = await request_body(client, make_project, tmp_path, producer)
    original_assets = {asset.id for asset in (await snapshot())[1]}
    handler = CountingHandler()
    monkeypatch.setitem(registry._HANDLERS, body["capability_id"], handler)
    pool = FakePool(mode, state)
    install_queue(monkeypatch, pool)
    response = await client.post(f"{API}/jobs", json=body)
    assert response.status_code == 500 and response.json() == ERROR_BODY
    jobs, assets = await snapshot()
    assert len(jobs) == 1 and {asset.id for asset in assets} == original_assets
    assert handler.calls == 0
    job = jobs[0]
    assert job.status == JobStatus.failed and job.error == svc.DISPATCH_FAILURE_MESSAGE
    assert job.progress == 0 and job.output is None and job.started_at is None
    assert job.finished_at is not None and job.project_id == uuid.UUID(pid)
    assert job.created_at is not None and job.updated_at is not None
    assert pool.committed_before_enqueue and pool.closed and len(pool.calls) == 1
    if producer == "export":
        assert (
            job.input["execution_snapshot"]["composition_revision_id"]
            == body["input"]["composition_revision_id"]
        )
    detail = await client.get(f"{API}/jobs/{job.id}")
    assert detail.json()["status"] == "failed"
    assert detail.json()["error"] == svc.DISPATCH_FAILURE_MESSAGE
    failed = (await client.get(f"{API}/projects/{pid}/jobs?status=failed")).json()
    queued = (await client.get(f"{API}/projects/{pid}/jobs?status=queued")).json()
    assert failed["total"] == 1 and queued["total"] == 0
    assert failed["items"][0]["id"] == str(job.id)
    assert await run_job({}, str(job.id)) == "failed"
    assert handler.calls == 0 and {asset.id for asset in (await snapshot())[1]} == original_assets
    assert (await client.post(f"{API}/jobs/{job.id}/cancel")).status_code == 422


async def test_pool_creation_failure_and_raw_enqueuer_exception(client, make_project, monkeypatch):
    pid = await make_project()

    async def unavailable(*_args):
        raise ConnectionError(RAW_DIAGNOSTIC)

    monkeypatch.setattr(queue, "get_arq_pool", unavailable)
    app.dependency_overrides[get_enqueuer] = lambda: queue.enqueue_job
    for enqueuer in [queue.enqueue_job, unavailable]:
        app.dependency_overrides[get_enqueuer] = lambda enqueue=enqueuer: enqueue
        response = await client.post(f"{API}/jobs", json={
            "capability_id": "media.titlecard", "project_id": pid, "input": {"text": "Test"},
        })
        assert response.status_code == 500 and response.json() == ERROR_BODY
    jobs, assets = await snapshot()
    assert len(jobs) == 2 and assets == []
    assert all(job.status == JobStatus.failed for job in jobs)


@pytest.mark.parametrize("mode,state,expected", [
    ("accepted", "queued", DispatchOutcome.ACCEPTED),
    ("close_error", "queued", DispatchOutcome.ACCEPTED),
    ("existing", "queued", DispatchOutcome.ALREADY_DISPATCHED),
    ("existing", "deferred", DispatchOutcome.ALREADY_DISPATCHED),
    ("existing", "in_progress", DispatchOutcome.ALREADY_DISPATCHED),
    ("lost_ack", "queued", DispatchOutcome.ALREADY_DISPATCHED),
])
async def test_acknowledged_or_existing_dispatch_stays_runnable(
    client, make_project, monkeypatch, mode, state, expected,
):
    pid = await make_project()
    pool = FakePool(mode, state)
    install_queue(monkeypatch, pool)
    outcomes = []

    async def recording_enqueue(identity):
        outcome = await queue.enqueue_job(identity)
        outcomes.append(outcome)
        return outcome

    app.dependency_overrides[get_enqueuer] = lambda: recording_enqueue
    response = await client.post(f"{API}/jobs", json={
        "capability_id": "media.titlecard", "project_id": pid, "input": {"text": "Test"},
    })
    assert response.status_code == 201
    assert outcomes == [expected]
    job_id = uuid.UUID(response.json()["id"])
    assert response.json()["status"] == "queued" and response.json()["error"] is None
    assert len(pool.tasks) == 1 and len(pool.calls) == 1 and pool.closed
    assert pool.calls[0] == (queue.RUN_JOB_TASK, (str(job_id),), {"_job_id": str(job_id)})
    # Duplicate enqueue does not require a second task or a different identity.
    pool.mode = "existing"
    assert await queue.enqueue_job(job_id) == DispatchOutcome.ALREADY_DISPATCHED
    assert len(pool.tasks) == 1
    handler = CountingHandler()
    monkeypatch.setitem(registry._HANDLERS, "media.titlecard", handler)
    assert await run_job({}, str(job_id)) == "succeeded"
    assert await run_job({}, str(job_id)) == "succeeded"
    jobs, assets = await snapshot()
    assert len(assets) == 1 and handler.calls == 1
    assert jobs[0].output["outputs"][0]["asset_id"] == str(assets[0].id)


@pytest.mark.parametrize("worker_status", list(JobStatus)[1:])
async def test_submission_error_preserves_worker_or_cancellation_winner(
    client, make_project, monkeypatch, worker_status,
):
    pid = await make_project()

    async def claimed_then_error(job_id):
        async with async_session_factory() as worker:
            assert await svc.claim_job(worker, job_id)
            if worker_status != JobStatus.running:
                job = await svc.get_job(worker, job_id)
                job.status = worker_status
                job.output = {"authoritative": True}
                job.error = "Existing terminal state"
                await worker.commit()
        raise ConnectionError(RAW_DIAGNOSTIC)

    app.dependency_overrides[get_enqueuer] = lambda: claimed_then_error
    response = await client.post(f"{API}/jobs", json={
        "capability_id": "media.titlecard", "project_id": pid, "input": {"text": "Test"},
    })
    assert response.status_code == 201 and response.json()["status"] == worker_status.value
    jobs, _ = await snapshot()
    assert jobs[0].status == worker_status
    if worker_status != JobStatus.running:
        assert jobs[0].output == {"authoritative": True}
        assert jobs[0].error == "Existing terminal state"


@pytest.mark.parametrize("iteration", range(4))
async def test_claim_and_compensation_have_exactly_one_winner(make_project, iteration):
    pid = uuid.UUID(await make_project())
    async with async_session_factory() as session:
        job = await svc.create_job(session, JobCreate(
            capability_id="media.titlecard", project_id=pid, input={"text": "Test"},
        ))
        job_id = job.id
    ready = asyncio.Event()
    arrivals = 0

    async def compete(operation):
        nonlocal arrivals
        async with async_session_factory() as session:
            arrivals += 1
            if arrivals == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 5)
            return await operation(session, job_id)

    claimed, compensated = await asyncio.gather(
        compete(svc.claim_job), compete(svc.fail_queued_dispatch),
    )
    assert int(claimed) + int(compensated) == 1
    jobs, assets = await snapshot()
    assert jobs[0].status == (JobStatus.running if claimed else JobStatus.failed)
    assert assets == []


async def test_compensation_conditional_sql_and_terminal_noop(make_project):
    pid = uuid.UUID(await make_project())
    async with async_session_factory() as session:
        job = await svc.create_job(session, JobCreate(
            capability_id="media.titlecard", project_id=pid, input={"text": "Test"},
        ))
        job_id = job.id
        execute = session.execute
        statements = []

        async def recording(statement, *args, **kwargs):
            statements.append(statement)
            return await execute(statement, *args, **kwargs)

        session.execute = recording
        assert await svc.fail_queued_dispatch(session, job_id)
        for dialect in [sqlite.dialect(), postgresql.dialect()]:
            compiled = statements[0].compile(dialect=dialect)
            assert "jobs.id =" in str(compiled) and "jobs.status =" in str(compiled)
            assert "queued" in compiled.params.values()
        assert not await svc.fail_queued_dispatch(session, job_id)


@pytest.mark.parametrize("phase", ["execute", "commit"])
async def test_compensation_database_failure_is_rolled_back_and_sanitized(
    client, make_project, monkeypatch, phase,
):
    pid = await make_project()
    original = svc.fail_queued_dispatch

    async def broken_compensation(session, job_id):
        async def fail(*_args, **_kwargs):
            raise RuntimeError(RAW_DIAGNOSTIC)
        with monkeypatch.context() as patch:
            patch.setattr(session, phase, fail)
            try:
                return await original(session, job_id)
            finally:
                assert not session.in_transaction()

    monkeypatch.setattr(svc, "fail_queued_dispatch", broken_compensation)
    pool = FakePool("raise")
    install_queue(monkeypatch, pool)
    response = await client.post(f"{API}/jobs", json={
        "capability_id": "media.titlecard", "project_id": pid, "input": {"text": "Test"},
    })
    assert response.status_code == 500 and response.json() == ERROR_BODY
    jobs, assets = await snapshot()
    # Without a successful compensation commit, durable dispatch needs an outbox.
    assert jobs[0].status == JobStatus.queued and assets == []


async def test_agent_approval_compensation_is_unchanged(client, make_project, monkeypatch):
    _, _, plan_id = await _create_plan(client, make_project, monkeypatch)
    pool = FakePool("raise")
    install_queue(monkeypatch, pool)
    app.dependency_overrides[get_plan_enqueuer] = lambda: queue.enqueue_plan
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test",
    ) as http:
        response = await http.post(f"{API}/agent/plans/{plan_id}/approve")
    assert response.status_code == 500
    async with async_session_factory() as session:
        plan = await session.get(AgentPlan, uuid.UUID(plan_id))
        attempts = list(await session.scalars(select(AgentPlanAttempt)))
        assert plan.status.value == "proposed" and plan.active_attempt_id is None
        assert attempts[0].status == "failed"
        assert attempts[0].error == "Execution could not be enqueued."
        assert list(await session.scalars(select(Job))) == []


async def test_dispatch_helper_controlled_error(make_project):
    pid = uuid.UUID(await make_project())
    async with async_session_factory() as session:
        job = await svc.create_job(session, JobCreate(
            capability_id="media.titlecard", project_id=pid, input={"text": "Test"},
        ))

        async def rejected(_identity):
            return DispatchOutcome.NOT_DISPATCHED

        with pytest.raises(AppError, match=svc.DISPATCH_FAILURE_MESSAGE):
            await svc.dispatch_job(session, job.id, rejected)


async def test_inspection_timeout_is_unknown(monkeypatch):
    pool = FakePool("existing")

    async def stalled(*_args):
        await asyncio.Event().wait()

    monkeypatch.setattr(pool, "zscore", stalled)
    monkeypatch.setattr(queue, "DISPATCH_INSPECTION_TIMEOUT", 0.01)
    assert await queue._existing_job_dispatch(pool, uuid.uuid4()) == DispatchOutcome.UNKNOWN


async def test_existing_task_must_match_worker_function_and_job_id():
    pool = FakePool("existing")
    identity = uuid.uuid4()
    for function, args in [(queue.RUN_PLAN_TASK, (str(identity),)),
                           (queue.RUN_JOB_TASK, (str(uuid.uuid4()),))]:
        pool.tasks[str(identity)] = serialize_job(function, args, {}, None, timestamp_ms())
        assert (
            await queue._existing_job_dispatch(pool, identity)
            == DispatchOutcome.NOT_DISPATCHED
        )


async def test_worker_can_finish_after_dispatch_compensation_loses(
    client, make_project, monkeypatch,
):
    pid = await make_project()
    handler = CountingHandler()
    handler.release.clear()
    monkeypatch.setitem(registry._HANDLERS, "media.titlecard", handler)
    workers = []

    async def claimed_then_lost_ack(identity):
        workers.append(asyncio.create_task(run_job({}, str(identity))))
        await asyncio.wait_for(handler.entered.wait(), 5)
        raise TimeoutError(RAW_DIAGNOSTIC)

    app.dependency_overrides[get_enqueuer] = lambda: claimed_then_lost_ack
    try:
        response = await client.post(f"{API}/jobs", json={
            "capability_id": "media.titlecard", "project_id": pid, "input": {"text": "Test"},
        })
        assert response.status_code == 201 and response.json()["status"] == "running"
        assert response.json()["error"] is None and (await snapshot())[1] == []
    finally:
        handler.release.set()
        await asyncio.gather(*workers)
    jobs, assets = await snapshot()
    assert jobs[0].status == JobStatus.succeeded and jobs[0].error is None
    assert len(assets) == 1 and handler.calls == 1
    assert await run_job({}, str(jobs[0].id)) == "succeeded"
    assert len((await snapshot())[1]) == 1 and handler.calls == 1
