"""Task #19B: recover Jobs and Agent plans a dead worker left running or queued."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

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
    PLAN_TASK_UNAVAILABLE,
    ScanCursor,
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


async def _plans_unanswered(pairs):
    return {}          # every approved attempt stays UNKNOWN


async def _reconcile(**kwargs) -> dict[str, int]:
    kwargs.setdefault("inspect", _no_queued)
    kwargs.setdefault("inspect_plans", _plans_unanswered)
    kwargs.setdefault("plan_cursor", ScanCursor())
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
    for message in (JOB_INTERRUPTED, PLAN_INTERRUPTED, PLAN_TASK_UNAVAILABLE):
        assert "/" not in message and "redis" not in message.lower()
        assert "arq" not in message.lower()
    assert timedelta(minutes=10) == reconciliation.QUEUED_GRACE


# --- approved plans whose task is gone (Task #21B) ---------------------------------------

GRACE = timedelta(minutes=10)
AGED = NOW - GRACE - timedelta(seconds=1)
YOUNG = NOW - GRACE + timedelta(seconds=60)


async def _approved(project_id: str, *, created: datetime, plan_status=AgentPlanStatus.approved,
                    attempt_status: str = "approved", active: bool = True,
                    attempt_number: int = 1) -> tuple[uuid.UUID, uuid.UUID]:
    async with async_session_factory() as session:
        agent_session = AgentSession(project_id=uuid.UUID(project_id), asset_ids=[])
        session.add(agent_session)
        await session.flush()
        plan = AgentPlan(session_id=agent_session.id, status=plan_status,
                         steps=[], step_runs=[], result_asset_ids=[])
        session.add(plan)
        await session.flush()
        attempt = AgentPlanAttempt(plan_id=plan.id, attempt_number=attempt_number,
                                   status=attempt_status, job_ids=[], result_asset_ids=[],
                                   step_runs=[], created_at=created)
        session.add(attempt)
        await session.flush()
        if active:
            plan.active_attempt_id = attempt.id
        await session.commit()
        return plan.id, attempt.id


def _answer(outcome: DispatchOutcome, seen: list | None = None):
    async def inspect(pairs):
        if seen is not None:
            seen.extend(pairs)
        return dict.fromkeys(pairs, outcome)
    return inspect


async def _plans_never_inspected(pairs):
    raise AssertionError(f"plan queue inspected for {pairs}")


async def _reconcile_plans(inspect_plans, cursor: ScanCursor | None = None) -> dict[str, int]:
    return await reconcile_all(async_session_factory, now=NOW, inspect=_no_queued,
                               inspect_plans=inspect_plans, plan_cursor=cursor or ScanCursor())


async def _assert_untouched(plan_id, attempt_id, plan_status=AgentPlanStatus.approved,
                            attempt_status="approved") -> None:
    plan, attempt = await _plan_state(plan_id, attempt_id)
    assert plan.status == plan_status and plan.error is None
    if plan_status == AgentPlanStatus.approved:
        assert plan.active_attempt_id == attempt_id
    assert attempt.status == attempt_status and attempt.finished_at is None


async def test_an_approved_plan_whose_task_is_gone_fails(make_project) -> None:
    pid = await make_project("approved-gone")
    plan_id, attempt_id = await _approved(pid, created=AGED)
    seen: list = []
    counts = await _reconcile_plans(_answer(DispatchOutcome.NOT_DISPATCHED, seen))
    assert seen == [(plan_id, attempt_id)]
    assert counts["approved_plans"] == 1 and counts["approved_uncertain"] == 0
    plan, attempt = await _plan_state(plan_id, attempt_id)
    assert plan.status == AgentPlanStatus.failed and plan.error == PLAN_TASK_UNAVAILABLE
    assert plan.active_attempt_id is None
    assert attempt.status == "failed" and attempt.error == PLAN_TASK_UNAVAILABLE
    assert attempt.finished_at is not None and attempt.started_at is None
    async with async_session_factory() as session:          # nothing was published
        assert await session.scalar(select(func.count()).select_from(Job)) == 0
        assert await session.scalar(select(func.count()).select_from(Asset)) == 0
    # Idempotent: no candidate is left, so the queue is not even asked.
    assert (await _reconcile_plans(_plans_never_inspected))["approved_plans"] == 0


async def test_young_approved_plans_are_not_inspected(make_project) -> None:
    pid = await make_project("approved-young")
    plan_id, attempt_id = await _approved(pid, created=YOUNG)
    assert (await _reconcile_plans(_plans_never_inspected))["approved_plans"] == 0
    await _assert_untouched(plan_id, attempt_id)


@pytest.mark.parametrize("outcome", [DispatchOutcome.ALREADY_DISPATCHED,
                                     DispatchOutcome.UNKNOWN])
async def test_a_held_or_uncertain_task_keeps_the_plan_approved(make_project, outcome) -> None:
    pid = await make_project(f"approved-{outcome.value}")
    plan_id, attempt_id = await _approved(pid, created=AGED - timedelta(days=3))
    counts = await _reconcile_plans(_answer(outcome))
    assert counts["approved_plans"] == 0
    assert counts["approved_uncertain"] == (1 if outcome == DispatchOutcome.UNKNOWN else 0)
    await _assert_untouched(plan_id, attempt_id)


async def test_an_inspector_that_answers_nothing_is_uncertain(make_project) -> None:
    pid = await make_project("approved-silent")
    plan_id, attempt_id = await _approved(pid, created=AGED)
    counts = await _reconcile_plans(_plans_unanswered)
    assert counts["approved_plans"] == 0 and counts["approved_uncertain"] == 1
    await _assert_untouched(plan_id, attempt_id)


async def test_only_the_absent_projects_plan_changes(make_project) -> None:
    gone = await _approved(await make_project("approved-iso-a"), created=AGED)
    held = await _approved(await make_project("approved-iso-b"), created=AGED)

    async def inspect(pairs):
        return {pair: DispatchOutcome.NOT_DISPATCHED if pair == gone
                else DispatchOutcome.ALREADY_DISPATCHED for pair in pairs}

    assert (await _reconcile_plans(inspect))["approved_plans"] == 1
    plan, _attempt = await _plan_state(*gone)
    assert plan.status == AgentPlanStatus.failed
    await _assert_untouched(*held)


@pytest.mark.parametrize("case", ["terminal-succeeded", "terminal-failed", "running",
                                  "attempt-running", "attempt-failed"])
async def test_only_a_fully_approved_pair_is_a_candidate(make_project, case) -> None:
    plan_status, attempt_status = {
        "terminal-succeeded": (AgentPlanStatus.succeeded, "approved"),
        "terminal-failed": (AgentPlanStatus.failed, "approved"),
        "running": (AgentPlanStatus.running, "approved"),
        "attempt-running": (AgentPlanStatus.approved, "running"),
        "attempt-failed": (AgentPlanStatus.approved, "failed"),
    }[case]
    pid = await make_project(f"approved-{case}")
    plan_id, attempt_id = await _approved(pid, created=AGED, plan_status=plan_status,
                                          attempt_status=attempt_status)
    await _reconcile_plans(_plans_never_inspected)
    plan, attempt = await _plan_state(plan_id, attempt_id)
    assert plan.status == plan_status and attempt.status == attempt_status
    assert plan.error is None and attempt.error is None


async def test_a_legacy_plan_without_an_active_attempt_is_excluded(make_project) -> None:
    pid = await make_project("approved-legacy")
    plan_id, attempt_id = await _approved(pid, created=AGED, active=False)
    async with async_session_factory() as session:      # legacy: no attempt rows at all
        legacy = AgentPlan(session_id=(await session.get(AgentPlan, plan_id)).session_id,
                           status=AgentPlanStatus.approved, steps=[], step_runs=[],
                           result_asset_ids=[])
        session.add(legacy)
        await session.commit()
        legacy_id = legacy.id
    await _reconcile_plans(_plans_never_inspected)
    plan, attempt = await _plan_state(plan_id, attempt_id)
    assert plan.status == AgentPlanStatus.approved and attempt.status == "approved"
    async with async_session_factory() as session:
        assert (await session.get(AgentPlan, legacy_id)).status == AgentPlanStatus.approved


async def test_a_plan_claim_during_inspection_wins(make_project) -> None:
    pid = await make_project("approved-claim-first")
    plan_id, attempt_id = await _approved(pid, created=AGED)

    async def claim_then_absent(pairs):
        async with async_session_factory() as session:
            assert await agent_svc.claim_plan_attempt(session, plan_id, attempt_id) is not None
        return dict.fromkeys(pairs, DispatchOutcome.NOT_DISPATCHED)

    assert (await _reconcile_plans(claim_then_absent))["approved_plans"] == 0
    plan, attempt = await _plan_state(plan_id, attempt_id)
    assert plan.status == AgentPlanStatus.running and plan.active_attempt_id == attempt_id
    assert attempt.status == "running" and attempt.error is None


async def test_a_task_delivered_after_recovery_does_nothing(make_project) -> None:
    pid = await make_project("approved-late-task")
    plan_id, attempt_id = await _approved(pid, created=AGED)
    await _reconcile_plans(_answer(DispatchOutcome.NOT_DISPATCHED))
    # A late enqueue succeeds; its delivery cannot claim the failed attempt.
    assert await tasks.run_plan({}, str(plan_id), str(attempt_id)) == "failed"
    # A late enqueue error tries to release the attempt; it is no longer approved.
    async with async_session_factory() as session:
        await agent_svc.release_unqueued_attempt(
            session, plan_id, attempt_id, AgentPlanStatus.proposed, None, "late")
    plan, attempt = await _plan_state(plan_id, attempt_id)
    assert plan.status == AgentPlanStatus.failed and plan.error == PLAN_TASK_UNAVAILABLE
    assert attempt.status == "failed" and attempt.error == PLAN_TASK_UNAVAILABLE
    assert attempt.started_at is None
    async with async_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Job)) == 0


async def test_a_newer_attempt_is_never_overwritten(make_project) -> None:
    pid = await make_project("approved-newer")
    plan_id, old_attempt = await _approved(pid, created=AGED)
    newer_id: list[uuid.UUID] = []

    async def released_and_reapproved(pairs):
        # Between selection and update the old attempt is released and the
        # plan approved again with a fresh attempt.
        async with async_session_factory() as session:
            await agent_svc.release_unqueued_attempt(
                session, plan_id, old_attempt, AgentPlanStatus.proposed, None, "release")
        async with async_session_factory() as session:
            _row, newer, _p, _e = await agent_svc.approve_plan(session, plan_id)
            newer_id.append(newer.id)
        return dict.fromkeys(pairs, DispatchOutcome.NOT_DISPATCHED)

    assert (await _reconcile_plans(released_and_reapproved))["approved_plans"] == 0
    plan, newer = await _plan_state(plan_id, newer_id[0])
    assert plan.status == AgentPlanStatus.approved and plan.active_attempt_id == newer.id
    assert newer.status == "approved" and newer.error is None


async def test_concurrent_reconcilers_recover_each_plan_once(make_project) -> None:
    pid = await make_project("approved-parallel")
    pairs = [await _approved(pid, created=AGED) for _ in range(6)]
    absent = _answer(DispatchOutcome.NOT_DISPATCHED)
    results = await asyncio.gather(*(_reconcile_plans(absent) for _ in range(3)))
    assert sum(result["approved_plans"] for result in results) == 6
    for plan_id, attempt_id in pairs:
        plan, attempt = await _plan_state(plan_id, attempt_id)
        assert plan.status == AgentPlanStatus.failed and attempt.status == "failed"


async def test_a_recovered_plan_is_approved_again_with_a_new_task(make_project) -> None:
    pid = await make_project("approved-retry")
    plan_id, old_attempt = await _approved(pid, created=AGED)
    await _reconcile_plans(_answer(DispatchOutcome.NOT_DISPATCHED))
    async with async_session_factory() as session:
        plan, new_attempt, previous, _error = await agent_svc.approve_plan(session, plan_id)
    assert previous == AgentPlanStatus.failed
    assert plan.status == AgentPlanStatus.approved and plan.error is None
    assert plan.active_attempt_id == new_attempt.id != old_attempt
    assert new_attempt.attempt_number == 2
    assert queue.plan_task_id(plan_id, new_attempt.id) != queue.plan_task_id(plan_id,
                                                                             old_attempt)
    # The abandoned attempt's task, delivered late, cannot take the new attempt.
    assert await tasks.run_plan({}, str(plan_id), str(old_attempt)) == "approved"
    plan, attempt = await _plan_state(plan_id, new_attempt.id)
    assert plan.status == AgentPlanStatus.approved and attempt.status == "approved"
    _plan_row, old = await _plan_state(plan_id, old_attempt)
    assert old.status == "failed" and old.error == PLAN_TASK_UNAVAILABLE   # history kept


async def test_earlier_outputs_stay_reusable_after_recovery(make_project, client) -> None:
    pid = await make_project("approved-reuse")
    plan_id, first = await _plan(pid, attempt_started=OLD)
    await _reconcile()                                   # attempt 1 interrupted -> failed
    upload = await client.post(f"/api/v1/projects/{pid}/assets",
                               files={"file": ("out.png", b"step-output", "image/png")})
    output_id = upload.json()["id"]
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(output_id))
        asset.technical_metadata_status = "available"
        session.add(Job(project_id=uuid.UUID(pid), capability_id="image.resize",
                        status=JobStatus.succeeded, input={}, progress=100,
                        finished_at=OLD, agent_plan_id=plan_id, agent_attempt_id=first,
                        execution_identity="e" * 64,
                        output={"outputs": [{"asset_id": output_id, "media_type": "image",
                                             "filename": "out.png"}]}))
        await session.commit()
    async with async_session_factory() as session:       # attempt 2: its task is lost
        _row, second, _p, _e = await agent_svc.approve_plan(session, plan_id)
        stored = await session.get(AgentPlanAttempt, second.id)
        stored.created_at = AGED
        await session.commit()
    assert (await _reconcile_plans(_answer(DispatchOutcome.NOT_DISPATCHED)))[
        "approved_plans"] == 1
    async with async_session_factory() as session:       # attempt 3 reuses attempt 1
        _row, third, _p, _e = await agent_svc.approve_plan(session, plan_id)
        found = await reuse_svc.find_reusable_output(
            session, plan_id=plan_id, current_attempt=third,
            project_id=uuid.UUID(pid), identity="e" * 64)
    assert third.attempt_number == 3
    assert found is not None and str(found.asset.id) == output_id
    assert found.attempt.id == first


async def test_a_full_batch_of_queued_plans_cannot_hide_a_later_orphan(make_project,
                                                                      monkeypatch) -> None:
    monkeypatch.setattr(reconciliation, "BATCH_SIZE", 2)
    pid = await make_project("approved-fairness")
    held = [await _approved(pid, created=AGED - timedelta(hours=2 - i)) for i in range(2)]
    orphan = await _approved(pid, created=AGED)
    batches: list[list] = []

    async def inspect(pairs):
        batches.append(list(pairs))
        return {pair: DispatchOutcome.NOT_DISPATCHED if pair == orphan
                else DispatchOutcome.ALREADY_DISPATCHED for pair in pairs}

    # Without the cursor every run would inspect the same two oldest attempts.
    fresh = await _reconcile_plans(inspect)
    assert batches[-1] == held and fresh["approved_plans"] == 0
    cursor = ScanCursor()
    assert (await _reconcile_plans(inspect, cursor))["approved_plans"] == 0
    assert batches[-1] == held and cursor.position is not None
    assert (await _reconcile_plans(inspect, cursor))["approved_plans"] == 1
    assert batches[-1] == [orphan] and cursor.position is None          # wrapped around
    plan, _attempt = await _plan_state(*orphan)
    assert plan.status == AgentPlanStatus.failed
    await _reconcile_plans(inspect, cursor)
    assert batches[-1] == held                           # the scan starts over
    for pair in held:
        await _assert_untouched(*pair)


async def test_one_failing_pass_does_not_block_the_others(make_project, monkeypatch) -> None:
    pid = await make_project("approved-pass-isolation")
    plan_id, attempt_id = await _approved(pid, created=AGED)
    running_plan, running_attempt = await _plan(pid, attempt_started=OLD)

    async def broken(*_args):
        raise RuntimeError("bug in one pass")

    monkeypatch.setattr(reconciliation, "recover_running_jobs", broken)
    monkeypatch.setattr(reconciliation, "recover_queued_jobs", broken)
    counts = await _reconcile_plans(_answer(DispatchOutcome.NOT_DISPATCHED))
    assert counts["plans"] == 1 and counts["approved_plans"] == 1
    assert counts["running_jobs"] == 0 and counts["queued_jobs"] == 0
    plan, _attempt = await _plan_state(running_plan, running_attempt)
    assert plan.status == AgentPlanStatus.failed and plan.error == PLAN_INTERRUPTED
    plan, _attempt = await _plan_state(plan_id, attempt_id)
    assert plan.status == AgentPlanStatus.failed and plan.error == PLAN_TASK_UNAVAILABLE


# --- plan queue inspection ----------------------------------------------------------------


class _FakeTask:
    """Stands in for arq's Job: answers status()/info() from a table."""

    table: dict[str, tuple] = {}
    created: list[str] = []

    def __init__(self, job_id, _pool, **_kwargs):
        self.job_id = job_id
        _FakeTask.created.append(job_id)

    async def status(self):
        status, _info = _FakeTask.table[self.job_id]
        if isinstance(status, BaseException):
            raise status
        if status == "hang":
            await asyncio.sleep(3600)
        return status

    async def info(self):
        return _FakeTask.table[self.job_id][1]


class _FakePool:
    default_queue_name = "arq:queue"
    job_deserializer = None
    closed = False

    async def aclose(self):
        _FakePool.closed = True


def _definition(plan_id, attempt_id, **changes):
    values = {"function": "run_plan", "args": (str(plan_id), str(attempt_id)), "kwargs": {}}
    values.update(changes)
    return SimpleNamespace(**values)


async def test_plan_inspection_reads_the_enqueued_task_identity(monkeypatch) -> None:
    from arq.jobs import JobStatus as ArqStatus

    enqueued: dict = {}

    class RecordingPool(_FakePool):
        async def enqueue_job(self, function, *args, _job_id):
            enqueued.update(function=function, args=args, job_id=_job_id)

    async def pool():
        return RecordingPool()

    monkeypatch.setattr(queue, "get_arq_pool", pool)
    monkeypatch.setattr(queue, "ArqJob", _FakeTask)
    plan_id, attempt_id = uuid.uuid4(), uuid.uuid4()
    await queue.enqueue_plan(plan_id, attempt_id)
    assert enqueued["job_id"] == queue.plan_task_id(plan_id, attempt_id)
    _FakeTask.created = []
    _FakeTask.table = {enqueued["job_id"]: (
        ArqStatus.queued,
        SimpleNamespace(function=enqueued["function"], args=enqueued["args"], kwargs={}))}
    outcome = await queue.inspect_plan_dispatches([(plan_id, attempt_id)])
    assert outcome == {(plan_id, attempt_id): DispatchOutcome.ALREADY_DISPATCHED}
    assert _FakeTask.created == [enqueued["job_id"]]


async def test_plan_inspection_status_semantics(monkeypatch) -> None:
    from arq.jobs import JobResult
    from arq.jobs import JobStatus as ArqStatus

    async def pool():
        return _FakePool()

    monkeypatch.setattr(queue, "get_arq_pool", pool)
    monkeypatch.setattr(queue, "ArqJob", _FakeTask)
    cases: dict[str, tuple] = {}
    expected: dict[tuple, DispatchOutcome] = {}

    def add(name, status, info_factory, outcome):
        pair = (uuid.uuid4(), uuid.uuid4())
        info = info_factory(*pair) if info_factory else None
        cases[queue.plan_task_id(*pair)] = (status, info)
        expected[pair] = outcome

    present, absent = DispatchOutcome.ALREADY_DISPATCHED, DispatchOutcome.NOT_DISPATCHED
    add("queued", ArqStatus.queued, _definition, present)
    add("deferred", ArqStatus.deferred, _definition, present)
    add("in progress", ArqStatus.in_progress, _definition, present)
    add("finished or failed result", ArqStatus.complete, None, absent)
    add("not found", ArqStatus.not_found, None, absent)
    add("queue entry without definition", ArqStatus.queued, None, absent)
    add("someone else's task", ArqStatus.queued,
        lambda p, a: _definition(p, a, function="run_job"), absent)
    add("other arguments", ArqStatus.in_progress,
        lambda p, a: _definition(p, a, args=(str(p),)), absent)
    add("a stored result as info", ArqStatus.in_progress,
        lambda p, a: JobResult(function="run_plan", args=(str(p), str(a)), kwargs={},
                               job_try=1, enqueue_time=NOW, score=None, success=False,
                               result=RuntimeError("x"), start_time=NOW, finish_time=NOW,
                               queue_name="arq:queue", job_id=queue.plan_task_id(p, a)),
        absent)
    add("lookup error", ConnectionError("lost"), None, DispatchOutcome.UNKNOWN)
    _FakeTask.table = cases
    _FakePool.closed = False
    assert await queue.inspect_plan_dispatches(list(expected)) == expected
    assert _FakePool.closed
    assert await queue.inspect_plan_dispatches([]) == {}


async def test_plan_inspection_is_bounded_by_one_deadline(monkeypatch) -> None:
    from arq.jobs import JobStatus as ArqStatus

    async def pool():
        return _FakePool()

    monkeypatch.setattr(queue, "get_arq_pool", pool)
    monkeypatch.setattr(queue, "ArqJob", _FakeTask)
    monkeypatch.setattr(queue, "PLAN_INSPECTION_BUDGET", 0.3)
    hung = [(uuid.uuid4(), uuid.uuid4()) for _ in range(30)]
    answered = (uuid.uuid4(), uuid.uuid4())
    _FakeTask.table = {queue.plan_task_id(*pair): ("hang", None) for pair in hung}
    _FakeTask.table[queue.plan_task_id(*answered)] = (ArqStatus.not_found, None)
    started = time.monotonic()
    outcomes = await queue.inspect_plan_dispatches([answered, *hung])
    assert time.monotonic() - started < 2          # not 30 sequential waits
    assert outcomes[answered] == DispatchOutcome.NOT_DISPATCHED
    assert all(outcomes[pair] == DispatchOutcome.UNKNOWN for pair in hung)


async def test_an_unreachable_plan_queue_is_uncertain(monkeypatch) -> None:
    async def down():
        raise ConnectionError("redis down")

    monkeypatch.setattr(queue, "get_arq_pool", down)
    pairs = [(uuid.uuid4(), uuid.uuid4())]
    assert await queue.inspect_plan_dispatches(pairs) == dict.fromkeys(
        pairs, DispatchOutcome.UNKNOWN)


# --- bounded pool acquisition (Task #21D) -------------------------------------------------

SAFETY = 5.0     # a hang fails the test instead of blocking it


def _hanging_pool(record: list[str]):
    async def pool():
        record.append("opening")
        try:
            await asyncio.Event().wait()           # PING that never returns
        except asyncio.CancelledError:
            record.append("cancelled")
            raise
    return pool


def _slow_pool(seconds: float):
    async def pool():
        await asyncio.sleep(seconds)
        return _FakePool()
    return pool


class _CountingTask(_FakeTask):
    cancelled = 0

    async def status(self):
        try:
            return await super().status()
        except asyncio.CancelledError:
            _CountingTask.cancelled += 1
            raise


def _lookups(answered: int, hung: int):
    from arq.jobs import JobStatus as ArqStatus

    fast = [(uuid.uuid4(), uuid.uuid4()) for _ in range(answered)]
    slow = [(uuid.uuid4(), uuid.uuid4()) for _ in range(hung)]
    _FakeTask.table = {queue.plan_task_id(*pair): (ArqStatus.not_found, None) for pair in fast}
    _FakeTask.table.update({queue.plan_task_id(*pair): ("hang", None) for pair in slow})
    _CountingTask.cancelled = 0
    return fast, slow


async def _timed(coro):
    started = time.monotonic()
    result = await asyncio.wait_for(coro, SAFETY)
    return result, time.monotonic() - started


async def test_plan_inspection_never_waits_for_a_silent_pool(monkeypatch) -> None:
    record: list[str] = []
    monkeypatch.setattr(queue, "get_arq_pool", _hanging_pool(record))
    monkeypatch.setattr(queue, "PLAN_INSPECTION_BUDGET", 0.4)
    pairs = [(uuid.uuid4(), uuid.uuid4()) for _ in range(3)]
    outcomes, elapsed = await _timed(queue.inspect_plan_dispatches(pairs))
    assert outcomes == dict.fromkeys(pairs, DispatchOutcome.UNKNOWN)
    assert elapsed < 1.0 and record == ["opening", "cancelled"]


async def test_plan_inspection_pool_slower_than_the_budget_is_uncertain(monkeypatch) -> None:
    monkeypatch.setattr(queue, "get_arq_pool", _slow_pool(2.0))
    monkeypatch.setattr(queue, "ArqJob", _FakeTask)
    monkeypatch.setattr(queue, "PLAN_INSPECTION_BUDGET", 1.0)
    monkeypatch.setattr(queue, "PLAN_INSPECTION_SETTLE", 0.1)
    fast, _slow = _lookups(answered=2, hung=0)
    outcomes, elapsed = await _timed(queue.inspect_plan_dispatches(fast))
    assert outcomes == dict.fromkeys(fast, DispatchOutcome.UNKNOWN)
    assert elapsed < 1.3


@pytest.mark.parametrize("pool_seconds", [0.6, 0.8])
async def test_lookups_only_get_the_time_left_after_the_pool(monkeypatch,
                                                            pool_seconds) -> None:
    # 0.6: most of the budget; 0.8: just before the 0.9 s lookup cutoff.
    monkeypatch.setattr(queue, "get_arq_pool", _slow_pool(pool_seconds))
    monkeypatch.setattr(queue, "ArqJob", _CountingTask)
    monkeypatch.setattr(queue, "PLAN_INSPECTION_BUDGET", 1.0)
    monkeypatch.setattr(queue, "PLAN_INSPECTION_SETTLE", 0.1)
    fast, slow = _lookups(answered=3, hung=5)
    outcomes, elapsed = await _timed(queue.inspect_plan_dispatches([*fast, *slow]))
    assert all(outcomes[pair] == DispatchOutcome.NOT_DISPATCHED for pair in fast)
    assert all(outcomes[pair] == DispatchOutcome.UNKNOWN for pair in slow)
    assert elapsed < 1.3                   # not pool time + a fresh full budget
    assert _CountingTask.cancelled == 5    # every unanswered lookup was cancelled


async def test_no_lookup_outlives_the_inspection(monkeypatch) -> None:
    monkeypatch.setattr(queue, "get_arq_pool", _slow_pool(0))
    monkeypatch.setattr(queue, "ArqJob", _CountingTask)
    monkeypatch.setattr(queue, "PLAN_INSPECTION_BUDGET", 0.4)
    _fast, slow = _lookups(answered=0, hung=25)
    _FakePool.closed = False
    outcomes, _elapsed = await _timed(queue.inspect_plan_dispatches(slow))
    assert set(outcomes.values()) == {DispatchOutcome.UNKNOWN}
    # 20 were inside a lookup (the concurrency bound); the other 5 were still
    # waiting for a slot. All 25 are gone.
    assert _CountingTask.cancelled == queue.PLAN_INSPECTION_CONCURRENCY == 20
    assert _FakePool.closed
    assert asyncio.all_tasks() == {asyncio.current_task()}


async def test_cancelling_the_inspection_stops_its_lookups(monkeypatch) -> None:
    # Worker shutdown: the cancellation propagates, lookups stop, pool closes.
    monkeypatch.setattr(queue, "get_arq_pool", _slow_pool(0))
    monkeypatch.setattr(queue, "ArqJob", _CountingTask)
    monkeypatch.setattr(queue, "PLAN_INSPECTION_BUDGET", 30.0)
    _fast, slow = _lookups(answered=0, hung=4)
    _FakePool.closed = False
    inspection = asyncio.create_task(queue.inspect_plan_dispatches(slow))
    await asyncio.sleep(0.1)
    inspection.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(inspection, SAFETY)
    assert _CountingTask.cancelled == 4 and _FakePool.closed
    assert asyncio.all_tasks() == {asyncio.current_task()}


async def test_empty_plan_inspection_does_not_open_redis(monkeypatch) -> None:
    async def unexpected():
        raise AssertionError("pool opened")

    monkeypatch.setattr(queue, "get_arq_pool", unexpected)
    assert await queue.inspect_plan_dispatches([]) == {}
    assert await queue.inspect_job_dispatches([]) == {}


async def test_job_inspection_never_waits_for_a_silent_pool(monkeypatch) -> None:
    record: list[str] = []
    monkeypatch.setattr(queue, "get_arq_pool", _hanging_pool(record))
    monkeypatch.setattr(queue, "INSPECTION_POOL_TIMEOUT", 0.3)
    ids = [uuid.uuid4(), uuid.uuid4()]
    outcomes, elapsed = await _timed(queue.inspect_job_dispatches(ids))
    assert outcomes == dict.fromkeys(ids, DispatchOutcome.UNKNOWN)
    assert elapsed < 1.0 and record == ["opening", "cancelled"]


async def test_job_inspection_semantics_are_unchanged(monkeypatch) -> None:
    from arq.jobs import JobStatus as ArqStatus

    async def pool():
        return _FakePool()

    monkeypatch.setattr(queue, "get_arq_pool", pool)
    monkeypatch.setattr(queue, "ArqJob", _FakeTask)
    held, gone, finished, foreign, broken = (uuid.uuid4() for _ in range(5))
    _FakeTask.table = {
        str(held): (ArqStatus.queued,
                    SimpleNamespace(function="run_job", args=(str(held),), kwargs={})),
        str(gone): (ArqStatus.not_found, None),
        str(finished): (ArqStatus.complete, None),
        str(foreign): (ArqStatus.queued,
                       SimpleNamespace(function="run_plan", args=(str(foreign),), kwargs={})),
        str(broken): (ConnectionError("lost"), None),
    }
    assert await queue.inspect_job_dispatches([held, gone, finished, foreign, broken]) == {
        held: DispatchOutcome.ALREADY_DISPATCHED,
        gone: DispatchOutcome.NOT_DISPATCHED,
        finished: DispatchOutcome.NOT_DISPATCHED,
        foreign: DispatchOutcome.NOT_DISPATCHED,
        broken: DispatchOutcome.UNKNOWN,
    }


async def test_a_silent_redis_endpoint_cannot_stall_recovery(monkeypatch) -> None:
    """The #21C reproduction: a real TCP endpoint that accepts and never replies."""
    from arq.connections import RedisSettings

    connected = asyncio.Event()
    closed = asyncio.Event()

    async def silent(reader, writer):
        connected.set()
        while await reader.read(1024):          # swallow PING, never answer
            pass
        closed.set()                             # the client hung up
        writer.close()

    server = await asyncio.start_server(silent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(queue, "redis_settings",
                        lambda: RedisSettings(host="127.0.0.1", port=port))
    monkeypatch.setattr(queue, "PLAN_INSPECTION_BUDGET", 0.6)
    monkeypatch.setattr(queue, "INSPECTION_POOL_TIMEOUT", 0.5)
    try:
        pairs = [(uuid.uuid4(), uuid.uuid4())]
        outcomes, elapsed = await _timed(queue.inspect_plan_dispatches(pairs))
        assert outcomes == dict.fromkeys(pairs, DispatchOutcome.UNKNOWN) and elapsed < 2
        assert connected.is_set()
        await asyncio.wait_for(closed.wait(), 2)          # no socket left open
        closed.clear()
        ids = [uuid.uuid4()]
        outcomes, elapsed = await _timed(queue.inspect_job_dispatches(ids))
        assert outcomes == dict.fromkeys(ids, DispatchOutcome.UNKNOWN) and elapsed < 2
        await asyncio.wait_for(closed.wait(), 2)
    finally:
        server.close()
        await server.wait_closed()


async def test_a_silent_queue_recovers_nothing(make_project, monkeypatch) -> None:
    pid = await make_project("approved-silent-redis")
    plan_id, attempt_id = await _approved(pid, created=AGED)
    job_id = await _job(pid, JobStatus.queued, created=NOW - timedelta(minutes=30))
    monkeypatch.setattr(queue, "get_arq_pool", _hanging_pool([]))
    monkeypatch.setattr(queue, "PLAN_INSPECTION_BUDGET", 0.3)
    monkeypatch.setattr(queue, "INSPECTION_POOL_TIMEOUT", 0.3)
    counts = await asyncio.wait_for(
        reconcile_all(async_session_factory, now=NOW, plan_cursor=ScanCursor()), SAFETY)
    assert counts["approved_plans"] == 0 and counts["approved_uncertain"] == 1
    assert counts["queued_jobs"] == 0 and counts["queued_uncertain"] == 1
    await _assert_untouched(plan_id, attempt_id)
    assert (await _get(job_id)).status == JobStatus.queued
