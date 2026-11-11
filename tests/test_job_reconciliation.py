"""Task #19B: recover Jobs and Agent plans a dead worker left running or queued."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.agent import executor
from app.agent import reuse as reuse_svc
from app.capabilities import registry
from app.capabilities.context import HandlerResult, OutputFile
from app.core.database import async_session_factory
from app.core.errors import ValidationError
from app.models.agent import AgentPlan, AgentPlanAttempt, AgentSession
from app.models.asset import Asset
from app.models.enums import AgentPlanStatus, JobStatus, MediaType
from app.models.job import Job
from app.services import agent as agent_svc
from app.services import jobs as jobs_svc
from app.services import reconciliation
from app.services.jobs import DISPATCH_FAILURE_MESSAGE
from app.services.reconciliation import (
    JOB_INTERRUPTED,
    PLAN_INTERRUPTED,
    reconcile_all,
    running_cutoff,
)
from app.workers import queue, tasks, worker
from app.workers.queue import WORKER_JOB_TIMEOUT_SECONDS, DispatchOutcome

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
LIMIT = timedelta(seconds=WORKER_JOB_TIMEOUT_SECONDS + 10 + 180)
OLD = NOW - LIMIT - timedelta(seconds=1)
RECENT = NOW - LIMIT + timedelta(seconds=60)   # long, but within the limit


async def _never_inspected(ids: list[uuid.UUID]) -> dict[uuid.UUID, DispatchOutcome]:
    raise AssertionError(f"queue inspected for {ids}")


async def _no_queued(ids: list[uuid.UUID]) -> dict[uuid.UUID, DispatchOutcome]:
    return {}


async def _job(project_id: str, status: JobStatus, *, started: datetime | None = None,
               created: datetime | None = None, capability: str = "media.titlecard") -> uuid.UUID:
    async with async_session_factory() as session:
        job = Job(project_id=uuid.UUID(project_id), capability_id=capability, status=status,
                  input={"text": "T", "duration": 1, "width": 64, "height": 48},
                  progress=40 if status == JobStatus.running else 0, started_at=started)
        if created is not None:
            job.created_at = created
        session.add(job)
        await session.commit()
        return job.id


async def _get(job_id: uuid.UUID) -> Job:
    async with async_session_factory() as session:
        job = await session.get(Job, job_id)
        assert job is not None
        return job


async def _reconcile(**kwargs) -> dict[str, int]:
    kwargs.setdefault("inspect", _no_queued)
    return await reconcile_all(async_session_factory, now=NOW, **kwargs)


# --- the bound ---------------------------------------------------------------------------


def test_bound_comes_from_the_worker_timeout() -> None:
    assert worker.WorkerSettings.job_timeout == WORKER_JOB_TIMEOUT_SECONDS
    assert running_cutoff(NOW) == NOW - timedelta(seconds=WORKER_JOB_TIMEOUT_SECONDS + 190)
    # No task overrides the timeout, so job_timeout bounds every run (arq uses
    # the largest function timeout for its in-progress key).
    for function in worker.WorkerSettings.functions:
        assert getattr(function, "timeout_s", None) is None


def test_recovery_runs_every_five_minutes_and_at_startup() -> None:
    crons = {job.coroutine.__name__: job for job in worker.WorkerSettings.cron_jobs}
    assert crons["reconcile_abandoned_work"].minute == set(range(0, 60, 5))
    assert crons["reconcile_abandoned_work"].unique is True
    assert crons["cleanup_temp_assets"].minute == 0       # unchanged
    assert worker.WorkerSettings.on_startup is worker.startup


async def test_startup_runs_recovery_and_never_crashes(monkeypatch) -> None:
    calls: list[str] = []

    async def failing(_factory):
        calls.append("run")
        raise ConnectionError("database down")

    monkeypatch.setattr(tasks, "reconcile_all", failing)
    await worker.startup({})                      # must not raise
    assert await tasks.reconcile_abandoned_work({}) == {}
    assert calls == ["run", "run"]


# --- running Jobs -------------------------------------------------------------------------


async def test_only_abandoned_running_jobs_fail(make_project) -> None:
    pid = await make_project("recover-running")
    old = await _job(pid, JobStatus.running, started=OLD)
    recent = await _job(pid, JobStatus.running, started=RECENT)
    finished = {status: await _job(pid, status, started=OLD) for status in
                (JobStatus.succeeded, JobStatus.cancelled, JobStatus.failed)}
    queued = await _job(pid, JobStatus.queued, created=NOW)
    counts = await _reconcile()
    assert counts["running_jobs"] == 1
    job = await _get(old)
    assert job.status == JobStatus.failed and job.error == JOB_INTERRUPTED
    assert job.finished_at is not None and job.progress == 40      # progress kept
    assert (await _get(recent)).status == JobStatus.running
    for status, job_id in finished.items():
        assert (await _get(job_id)).status == status
    assert (await _get(queued)).status == JobStatus.queued
    # Idempotent: nothing left to recover.
    assert (await _reconcile())["running_jobs"] == 0


async def test_concurrent_recoveries_do_not_conflict(make_project) -> None:
    pid = await make_project("recover-concurrent")
    ids = [await _job(pid, JobStatus.running, started=OLD) for _ in range(5)]
    results = await asyncio.gather(_reconcile(), _reconcile(), _reconcile())
    assert sum(result["running_jobs"] for result in results) == 5
    for job_id in ids:
        job = await _get(job_id)
        assert job.status == JobStatus.failed and job.error == JOB_INTERRUPTED


async def test_cancellation_and_recovery_race_safely(make_project) -> None:
    pid = await make_project("recover-cancel")
    cancelled_first = await _job(pid, JobStatus.running, started=OLD)
    async with async_session_factory() as session:
        await jobs_svc.cancel_job(session, cancelled_first)
    recovered_first = await _job(pid, JobStatus.running, started=OLD)
    await _reconcile()
    assert (await _get(cancelled_first)).status == JobStatus.cancelled
    async with async_session_factory() as session:
        with pytest.raises(ValidationError, match="already failed"):
            await jobs_svc.cancel_job(session, recovered_first)
    assert (await _get(recovered_first)).status == JobStatus.failed


class _BlockedHandler:
    """Finishes only after the test has recovered its Job (a 'stale' worker)."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, ctx):
        self.started.set()
        await self.release.wait()
        path = ctx.work_dir / "late.png"
        path.write_bytes(b"late output")
        return HandlerResult(outputs=[OutputFile(path, path.name, MediaType.image)])


async def test_a_stale_worker_cannot_publish_or_report_progress(make_project,
                                                                monkeypatch) -> None:
    pid = await make_project("recover-stale-worker")
    job_id = await _job(pid, JobStatus.queued)
    handler = _BlockedHandler()
    monkeypatch.setitem(registry._HANDLERS, "media.titlecard", handler)

    async def ready(_capability):
        return None

    monkeypatch.setattr(jobs_svc, "require_worker_readiness", ready)

    async def run():
        async with async_session_factory() as session:
            return await jobs_svc.execute_job(session, job_id)

    task = asyncio.create_task(run())
    await asyncio.wait_for(handler.started.wait(), 10)
    async with async_session_factory() as session:          # the worker "died" long ago
        job = await session.get(Job, job_id)
        job.started_at = OLD
        await session.commit()
    assert (await _reconcile())["running_jobs"] == 1
    await jobs_svc._progress_sink(job_id)(0.9)               # late progress is ignored
    handler.release.set()
    await asyncio.wait_for(task, 10)
    job = await _get(job_id)
    assert job.status == JobStatus.failed and job.error == JOB_INTERRUPTED
    assert job.output is None and job.progress != 100
    async with async_session_factory() as session:
        published = await session.scalar(select(func.count()).select_from(Asset).where(
            Asset.project_id == uuid.UUID(pid), Asset.source_job_id == job_id))
    assert published == 0


# --- queued Jobs --------------------------------------------------------------------------


async def test_queued_jobs_fail_only_when_definitely_gone(make_project) -> None:
    pid = await make_project("recover-queued")
    old = NOW - timedelta(minutes=11)
    gone = await _job(pid, JobStatus.queued, created=old)
    present = await _job(pid, JobStatus.queued, created=old)
    unknown = await _job(pid, JobStatus.queued, created=old)
    young = await _job(pid, JobStatus.queued, created=NOW - timedelta(minutes=9))
    seen: list[uuid.UUID] = []

    async def inspect(ids):
        seen.extend(ids)
        return {gone: DispatchOutcome.NOT_DISPATCHED,
                present: DispatchOutcome.ALREADY_DISPATCHED,
                unknown: DispatchOutcome.UNKNOWN}

    counts = await _reconcile(inspect=inspect)
    assert counts["queued_jobs"] == 1 and counts["queued_uncertain"] == 1
    assert young not in seen                                  # within the grace period
    job = await _get(gone)
    assert job.status == JobStatus.failed and job.error == DISPATCH_FAILURE_MESSAGE
    for job_id in (present, unknown, young):
        assert (await _get(job_id)).status == JobStatus.queued


async def test_a_claim_during_inspection_wins(make_project) -> None:
    pid = await make_project("recover-claim-race")
    job_id = await _job(pid, JobStatus.queued, created=NOW - timedelta(minutes=30))

    async def inspect(ids):
        async with async_session_factory() as session:      # a worker claims meanwhile
            assert await jobs_svc.claim_job(session, job_id)
        return dict.fromkeys(ids, DispatchOutcome.NOT_DISPATCHED)

    assert (await _reconcile(inspect=inspect))["queued_jobs"] == 0
    assert (await _get(job_id)).status == JobStatus.running


async def test_an_unreachable_queue_is_uncertain_not_absent(monkeypatch) -> None:
    async def down():
        raise ConnectionError("redis down")

    monkeypatch.setattr(queue, "get_arq_pool", down)
    ids = [uuid.uuid4(), uuid.uuid4()]
    assert await queue.inspect_job_dispatches(ids) == dict.fromkeys(ids, DispatchOutcome.UNKNOWN)


# --- Agent plans --------------------------------------------------------------------------


async def _plan(project_id: str, *, attempt_started: datetime) -> tuple[uuid.UUID, uuid.UUID]:
    async with async_session_factory() as session:
        agent_session = AgentSession(project_id=uuid.UUID(project_id), asset_ids=[])
        session.add(agent_session)
        await session.flush()
        plan = AgentPlan(session_id=agent_session.id, status=AgentPlanStatus.running,
                         steps=[], step_runs=[], result_asset_ids=[])
        session.add(plan)
        await session.flush()
        attempt = AgentPlanAttempt(plan_id=plan.id, attempt_number=1, status="running",
                                   started_at=attempt_started, job_ids=[],
                                   result_asset_ids=[], step_runs=[])
        session.add(attempt)
        await session.flush()
        plan.active_attempt_id = attempt.id
        await session.commit()
        return plan.id, attempt.id


async def _plan_state(plan_id: uuid.UUID, attempt_id: uuid.UUID):
    async with async_session_factory() as session:
        plan = await session.get(AgentPlan, plan_id)
        attempt = await session.get(AgentPlanAttempt, attempt_id)
        return plan, attempt


async def test_abandoned_plans_fail_and_can_be_retried(make_project) -> None:
    pid = await make_project("recover-plan")
    old_plan, old_attempt = await _plan(pid, attempt_started=OLD)
    live_plan, live_attempt = await _plan(pid, attempt_started=RECENT)
    assert (await _reconcile())["plans"] == 1
    plan, attempt = await _plan_state(old_plan, old_attempt)
    assert plan.status == AgentPlanStatus.failed and plan.error == PLAN_INTERRUPTED
    assert plan.active_attempt_id is None
    assert attempt.status == "failed" and attempt.finished_at is not None
    plan, attempt = await _plan_state(live_plan, live_attempt)
    assert plan.status == AgentPlanStatus.running and attempt.status == "running"
    assert (await _reconcile())["plans"] == 0                  # idempotent
    async with async_session_factory() as session:             # normal retry approval
        retried, new_attempt, _previous, _error = await agent_svc.approve_plan(session, old_plan)
    assert retried.status == AgentPlanStatus.approved
    assert retried.active_attempt_id == new_attempt.id != old_attempt


async def test_completed_steps_of_a_recovered_attempt_stay_reusable(make_project,
                                                                     client) -> None:
    pid = await make_project("recover-reuse")
    plan_id, attempt_id = await _plan(pid, attempt_started=OLD)
    upload = await client.post(f"/api/v1/projects/{pid}/assets",
                               files={"file": ("out.png", b"step-output", "image/png")})
    output_id = upload.json()["id"]
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(output_id))
        asset.technical_metadata_status = "available"
        session.add(Job(project_id=uuid.UUID(pid), capability_id="image.resize",
                        status=JobStatus.succeeded, input={}, progress=100,
                        finished_at=OLD, agent_plan_id=plan_id, agent_attempt_id=attempt_id,
                        execution_identity="e" * 64,
                        output={"outputs": [{"asset_id": output_id, "media_type": "image",
                                             "filename": "out.png"}]}))
        await session.commit()
    await _reconcile()
    async with async_session_factory() as session:
        _plan_row, new_attempt, _p, _e = await agent_svc.approve_plan(session, plan_id)
        found = await reuse_svc.find_reusable_output(
            session, plan_id=plan_id, current_attempt=new_attempt,
            project_id=uuid.UUID(pid), identity="e" * 64)
    assert found is not None and str(found.asset.id) == output_id


async def _executor_view(plan_id, attempt_id):
    session = async_session_factory()
    plan = await session.get(AgentPlan, plan_id)
    attempt = await session.get(AgentPlanAttempt, attempt_id)
    return session, plan, attempt


async def test_a_stale_executor_cannot_overwrite_recovery(make_project) -> None:
    pid = await make_project("recover-stale-executor")
    plan_id, attempt_id = await _plan(pid, attempt_started=OLD)
    session, plan_row, attempt = await _executor_view(plan_id, attempt_id)
    try:
        await _reconcile()                                     # recovery wins meanwhile
        plan_row.status = AgentPlanStatus.succeeded
        with pytest.raises(executor.StaleAttemptError):
            await executor._finish_attempt(session, plan_row, attempt, "succeeded", None, [])
        with pytest.raises(executor.StaleAttemptError):
            await executor._save_running_attempt(session, plan_row, attempt,
                                                 [{"step_id": "late"}], None)
    finally:
        await session.close()
    plan, stored_attempt = await _plan_state(plan_id, attempt_id)
    assert plan.status == AgentPlanStatus.failed and plan.error == PLAN_INTERRUPTED
    assert stored_attempt.status == "failed" and plan.step_runs == []


async def test_an_old_attempt_cannot_touch_a_newer_attempt(make_project) -> None:
    pid = await make_project("recover-newer-attempt")
    plan_id, old_attempt = await _plan(pid, attempt_started=OLD)
    await _reconcile()
    async with async_session_factory() as session:
        _row, newer, _p, _e = await agent_svc.approve_plan(session, plan_id)
        claimed = await agent_svc.claim_plan_attempt(session, plan_id, newer.id)
        assert claimed is not None
    # The old attempt's executor wakes up: its row is even made "running" again
    # to show that only the plan's active attempt may write.
    async with async_session_factory() as session:
        stale = await session.get(AgentPlanAttempt, old_attempt)
        stale.status = "running"
        await session.commit()
    session, plan_row, attempt = await _executor_view(plan_id, old_attempt)
    try:
        plan_row.status = AgentPlanStatus.succeeded
        plan_row.result_asset_ids = ["bogus"]
        with pytest.raises(executor.StaleAttemptError):
            await executor._finish_attempt(session, plan_row, attempt, "succeeded", None, [])
    finally:
        await session.close()
    async with async_session_factory() as session:
        plan = await session.get(AgentPlan, plan_id)
        assert plan.status == AgentPlanStatus.running
        assert plan.active_attempt_id == newer.id and plan.result_asset_ids == []


async def test_execute_plan_returns_quietly_when_it_lost_ownership(make_project,
                                                                   monkeypatch) -> None:
    pid = await make_project("recover-execute-plan")
    plan_id, attempt_id = await _plan(pid, attempt_started=OLD)

    async def claimed_then_recovered(session, plan, attempt):
        row = await session.get(AgentPlan, plan)
        stored = await session.get(AgentPlanAttempt, attempt)
        await reconcile_all(async_session_factory, now=NOW, inspect=_no_queued)
        return row, stored

    monkeypatch.setattr(agent_svc, "claim_plan_attempt", claimed_then_recovered)
    async with async_session_factory() as session:
        result = await executor.execute_plan(session, plan_id, attempt_id)
    assert result.status == AgentPlanStatus.failed and result.error == PLAN_INTERRUPTED


def test_messages_do_not_leak_internals() -> None:
    for message in (JOB_INTERRUPTED, PLAN_INTERRUPTED):
        assert "/" not in message and "redis" not in message.lower()
        assert "arq" not in message.lower()
    assert timedelta(minutes=10) == reconciliation.QUEUED_GRACE
