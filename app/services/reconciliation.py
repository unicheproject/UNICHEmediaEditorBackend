"""Recover work a dead worker left behind (SIGKILL, OOM, crash, host restart).

A Job claimed by a worker that then dies stays `running` forever: arq
redelivers its task, but `claim_job` only accepts `queued`, so nothing repairs
the database. The same holds for an Agent plan attempt, which then blocks
retry approval (only proposed/failed plans can be approved).

Why failing them is safe:
- arq wraps every task (run_job, run_plan) in asyncio.wait_for with the
  worker's job timeout and cancels it after that, and the cancellation path
  writes a terminal state. A Job's or attempt's `started_at` is set inside
  that task. So once started_at is older than the timeout plus a grace (arq
  keeps its in-progress key 10 s longer; 180 s more covers slow cancellation
  and clock differences), no live worker can still be running it.
- Every write here is a conditional update that rechecks the state, and
  workers publish, report progress and finish only while their row is still
  `running` (and, for plans, while they still own the active attempt). A
  late worker therefore cannot overwrite a recovery, and two recoveries
  cannot conflict.

Recovered work is failed, never retried automatically. Queued Jobs and
approved Agent attempts are failed only after a grace period and only when arq
definitely no longer holds their task; an unreachable or uncertain queue
leaves them untouched.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

from sqlalchemy import and_, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.models.agent import AgentPlan, AgentPlanAttempt
from app.models.enums import AgentPlanStatus, JobStatus
from app.models.job import Job
from app.services.jobs import DISPATCH_FAILURE_MESSAGE
from app.workers.queue import (
    WORKER_JOB_TIMEOUT_SECONDS,
    DispatchOutcome,
    PlanAttempt,
    inspect_job_dispatches,
    inspect_plan_dispatches,
)

logger = get_logger(__name__)

# arq keeps a task's in-progress key 10 s past the job timeout; 180 s more
# covers slow cancellation writes and clock differences between hosts.
ARQ_IN_PROGRESS_MARGIN_SECONDS = 10
RECOVERY_GRACE_SECONDS = 180
QUEUED_GRACE = timedelta(minutes=10)
BATCH_SIZE = 200

JOB_INTERRUPTED = "The worker stopped before this job finished. Please run it again."
PLAN_INTERRUPTED = "The worker stopped before this plan finished. Please run it again."
PLAN_TASK_UNAVAILABLE = (
    "The execution task was not available. Please approve the plan again."
)

DispatchInspector = Callable[[list[uuid.UUID]], Awaitable[dict[uuid.UUID, DispatchOutcome]]]
PlanDispatchInspector = Callable[
    [list[PlanAttempt]], Awaitable[dict[PlanAttempt, DispatchOutcome]]
]


@dataclass
class ScanCursor:
    """Where the next approved-attempt scan resumes, so a full batch of
    attempts that are legitimately still queued cannot hide later ones."""

    position: tuple[datetime, uuid.UUID] | None = None


# One per worker process; it starts again from the oldest after a restart.
APPROVED_PLAN_CURSOR = ScanCursor()


def running_cutoff(now: datetime) -> datetime:
    """Work started before this cannot belong to a live worker task."""
    limit = WORKER_JOB_TIMEOUT_SECONDS + ARQ_IN_PROGRESS_MARGIN_SECONDS + RECOVERY_GRACE_SECONDS
    return now - timedelta(seconds=limit)


def _rows(result: object) -> int:
    return cast(CursorResult, result).rowcount or 0


async def recover_running_jobs(session: AsyncSession, now: datetime) -> int:
    """running -> failed for Jobs whose worker task can no longer be alive."""
    cutoff = running_cutoff(now)
    ids = list(await session.scalars(
        select(Job.id)
        .where(Job.status == JobStatus.running, Job.started_at < cutoff)
        .order_by(Job.started_at)
        .limit(BATCH_SIZE)
    ))
    if not ids:
        await session.rollback()
        return 0
    result = await session.execute(
        update(Job)
        .where(Job.id.in_(ids), Job.status == JobStatus.running, Job.started_at < cutoff)
        .values(status=JobStatus.failed, error=JOB_INTERRUPTED, finished_at=now)
        .execution_options(synchronize_session=False)
    )
    await session.commit()
    return _rows(result)


async def recover_running_plans(session: AsyncSession, now: datetime) -> int:
    """Fail abandoned running attempts and release their plans for retry."""
    cutoff = running_cutoff(now)
    attempts = (await session.execute(
        select(AgentPlanAttempt.id, AgentPlanAttempt.plan_id)
        .where(AgentPlanAttempt.status == "running", AgentPlanAttempt.started_at < cutoff)
        .order_by(AgentPlanAttempt.started_at)
        .limit(BATCH_SIZE)
    )).all()
    recovered = 0
    for attempt_id, plan_id in attempts:
        # The plan changes only while this attempt still owns it.
        await session.execute(
            update(AgentPlan)
            .where(
                AgentPlan.id == plan_id,
                AgentPlan.status == AgentPlanStatus.running,
                AgentPlan.active_attempt_id == attempt_id,
            )
            .values(status=AgentPlanStatus.failed, error=PLAN_INTERRUPTED,
                    active_attempt_id=None)
            .execution_options(synchronize_session=False)
        )
        result = await session.execute(
            update(AgentPlanAttempt)
            .where(
                AgentPlanAttempt.id == attempt_id,
                AgentPlanAttempt.status == "running",
                AgentPlanAttempt.started_at < cutoff,
            )
            .values(status="failed", error=PLAN_INTERRUPTED, finished_at=now)
            .execution_options(synchronize_session=False)
        )
        recovered += _rows(result)
    await session.commit()
    return recovered


async def recover_queued_jobs(
    session_factory: async_sessionmaker[AsyncSession],
    now: datetime,
    inspect: DispatchInspector,
) -> tuple[int, int]:
    """queued -> failed only when arq definitely no longer holds the task.

    Returns (recovered, left_uncertain). No transaction is open while the
    queue is inspected.
    """
    cutoff = now - QUEUED_GRACE
    async with session_factory() as session:
        ids = list(await session.scalars(
            select(Job.id)
            .where(Job.status == JobStatus.queued, Job.created_at < cutoff)
            .order_by(Job.created_at)
            .limit(BATCH_SIZE)
        ))
    if not ids:
        return 0, 0
    outcomes = await inspect(ids)
    absent = [job_id for job_id in ids
              if outcomes.get(job_id) == DispatchOutcome.NOT_DISPATCHED]
    uncertain = sum(1 for job_id in ids
                    if outcomes.get(job_id, DispatchOutcome.UNKNOWN) == DispatchOutcome.UNKNOWN)
    if not absent:
        return 0, uncertain
    async with session_factory() as session:
        result = await session.execute(
            update(Job)
            .where(Job.id.in_(absent), Job.status == JobStatus.queued, Job.created_at < cutoff)
            .values(status=JobStatus.failed, error=DISPATCH_FAILURE_MESSAGE, finished_at=now)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
    return _rows(result), uncertain


async def recover_unqueued_plans(
    session_factory: async_sessionmaker[AsyncSession],
    now: datetime,
    inspect: PlanDispatchInspector,
    cursor: ScanCursor,
) -> tuple[int, int]:
    """approved -> failed for attempts whose run_plan task arq no longer holds.

    Candidates are a plan and its active attempt, both still approved, the
    attempt created more than QUEUED_GRACE ago. Plans without an active
    attempt (approved before attempts existed) have no known task id and are
    never candidates. Batches advance through the candidates with `cursor`
    and start over after the last one. Returns (recovered, left_uncertain).
    No transaction is open while the queue is inspected.
    """
    cutoff = now - QUEUED_GRACE
    query = (
        select(AgentPlanAttempt.plan_id, AgentPlanAttempt.id, AgentPlanAttempt.created_at)
        .join(AgentPlan, AgentPlan.id == AgentPlanAttempt.plan_id)
        .where(
            AgentPlanAttempt.status == "approved",
            AgentPlanAttempt.created_at < cutoff,
            AgentPlan.status == AgentPlanStatus.approved,
            AgentPlan.active_attempt_id == AgentPlanAttempt.id,
        )
    )
    if cursor.position is not None:
        after_created, after_id = cursor.position
        query = query.where(or_(
            AgentPlanAttempt.created_at > after_created,
            and_(AgentPlanAttempt.created_at == after_created, AgentPlanAttempt.id > after_id),
        ))
    async with session_factory() as session:
        rows = (await session.execute(
            query.order_by(AgentPlanAttempt.created_at, AgentPlanAttempt.id).limit(BATCH_SIZE)
        )).all()
    cursor.position = (rows[-1].created_at, rows[-1].id) if len(rows) == BATCH_SIZE else None
    if not rows:
        return 0, 0
    pairs = [(row.plan_id, row.id) for row in rows]
    outcomes = await inspect(pairs)
    uncertain = sum(1 for pair in pairs
                    if outcomes.get(pair, DispatchOutcome.UNKNOWN) == DispatchOutcome.UNKNOWN)
    recovered = 0
    for plan_id, attempt_id in pairs:
        if outcomes.get((plan_id, attempt_id)) != DispatchOutcome.NOT_DISPATCHED:
            continue
        # Plan first, then attempt: the same order as the worker's claim, so a
        # concurrent claim waits on (or wins) the plan row and only one side
        # changes the pair. Both change together or not at all.
        async with session_factory() as session:
            plan = await session.execute(
                update(AgentPlan)
                .where(
                    AgentPlan.id == plan_id,
                    AgentPlan.status == AgentPlanStatus.approved,
                    AgentPlan.active_attempt_id == attempt_id,
                )
                .values(status=AgentPlanStatus.failed, error=PLAN_TASK_UNAVAILABLE,
                        active_attempt_id=None)
                .execution_options(synchronize_session=False)
            )
            if _rows(plan) != 1:
                await session.rollback()
                continue
            attempt = await session.execute(
                update(AgentPlanAttempt)
                .where(
                    AgentPlanAttempt.id == attempt_id,
                    AgentPlanAttempt.status == "approved",
                    AgentPlanAttempt.created_at < cutoff,
                )
                .values(status="failed", error=PLAN_TASK_UNAVAILABLE, finished_at=now)
                .execution_options(synchronize_session=False)
            )
            if _rows(attempt) != 1:
                await session.rollback()
                continue
            await session.commit()
            recovered += 1
    return recovered, uncertain


async def reconcile_all(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    now: datetime | None = None,
    inspect: DispatchInspector = inspect_job_dispatches,
    inspect_plans: PlanDispatchInspector = inspect_plan_dispatches,
    plan_cursor: ScanCursor | None = None,
) -> dict[str, int]:
    """One bounded recovery run: running Jobs, running plans, queued Jobs,
    then approved plans. A pass that fails is logged and the others still run.
    """
    moment = now or datetime.now(UTC)
    cursor = APPROVED_PLAN_CURSOR if plan_cursor is None else plan_cursor
    counts = {"running_jobs": 0, "plans": 0, "queued_jobs": 0, "queued_uncertain": 0,
              "approved_plans": 0, "approved_uncertain": 0}

    async def running_jobs() -> None:
        async with session_factory() as session:
            counts["running_jobs"] = await recover_running_jobs(session, moment)

    async def running_plans() -> None:
        async with session_factory() as session:
            counts["plans"] = await recover_running_plans(session, moment)

    async def queued_jobs() -> None:
        counts["queued_jobs"], counts["queued_uncertain"] = await recover_queued_jobs(
            session_factory, moment, inspect)

    async def approved_plans() -> None:
        counts["approved_plans"], counts["approved_uncertain"] = await recover_unqueued_plans(
            session_factory, moment, inspect_plans, cursor)

    for name, recovery in (("running Jobs", running_jobs), ("running plans", running_plans),
                           ("queued Jobs", queued_jobs), ("approved plans", approved_plans)):
        try:
            await recovery()
        except Exception as exc:  # noqa: BLE001 - one failing pass must not block the others
            logger.warning("Recovery of %s did not run (%s)", name, type(exc).__name__)
    if any(counts.values()):
        logger.warning("Recovered abandoned work: %s", counts)
    return counts
