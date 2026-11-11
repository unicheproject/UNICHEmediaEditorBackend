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

Recovered work is failed, never retried automatically. Queued Jobs are
failed only after a grace period and only when arq definitely no longer
holds their task; an unreachable or uncertain queue leaves them untouched.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import cast

from sqlalchemy import select, update
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
    inspect_job_dispatches,
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

DispatchInspector = Callable[[list[uuid.UUID]], Awaitable[dict[uuid.UUID, DispatchOutcome]]]


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


async def reconcile_all(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    now: datetime | None = None,
    inspect: DispatchInspector = inspect_job_dispatches,
) -> dict[str, int]:
    """One bounded recovery pass: running Jobs, then plans, then queued Jobs."""
    moment = now or datetime.now(UTC)
    async with session_factory() as session:
        running_jobs = await recover_running_jobs(session, moment)
    async with session_factory() as session:
        plans = await recover_running_plans(session, moment)
    queued_jobs, uncertain = await recover_queued_jobs(session_factory, moment, inspect)
    counts = {"running_jobs": running_jobs, "plans": plans, "queued_jobs": queued_jobs,
              "queued_uncertain": uncertain}
    if running_jobs or plans or queued_jobs or uncertain:
        logger.warning("Recovered abandoned work: %s", counts)
    return counts
