"""arq queue connection + enqueue helper used by the API layer.

The API enqueues the opaque `run_job` task by id — it never references a
provider or executes work in the request handler.
"""

from __future__ import annotations

import asyncio
import uuid
from enum import Enum

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from arq.jobs import Job as ArqJob
from arq.jobs import JobResult
from arq.jobs import JobStatus as ArqJobStatus

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

RUN_JOB_TASK = "run_job"
RUN_PLAN_TASK = "run_plan"
RUN_ASSET_PROBE_TASK = "run_asset_probe"

# The arq worker's job timeout (WorkerSettings.job_timeout), the one source of
# truth: arq cancels every task (run_job, run_plan, cron) after this many
# seconds, which bounds how long a live worker can keep a Job running.
WORKER_JOB_TIMEOUT_SECONDS = 1800

# How long cancel_job() waits to hear back from arq before giving up (the DB
# is still updated immediately by services.jobs.cancel_job regardless).
CANCEL_TIMEOUT = 5.0
DISPATCH_INSPECTION_TIMEOUT = 5.0
# arq's create_pool sets only a connect timeout and then awaits PING with no
# read timeout, so a Redis that accepts connections but never answers would
# hang it for ever. Recovery inspection therefore always opens its pool under
# an explicit deadline (enough for arq's own connection retries).
INSPECTION_POOL_TIMEOUT = 10.0


class DispatchOutcome(Enum):
    ACCEPTED = "accepted"
    ALREADY_DISPATCHED = "already_dispatched"
    NOT_DISPATCHED = "not_dispatched"
    UNKNOWN = "unknown"


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def get_arq_pool() -> ArqRedis:
    return await create_pool(redis_settings())


async def _open_inspection_pool(timeout: float, purpose: str) -> ArqRedis | None:
    """The arq pool within `timeout` seconds, or None when Redis is unreachable,
    refuses or stays silent: an uncertain queue, never an empty one.

    Cancelling the pending PING makes redis-py close that connection, so a
    timeout leaves no socket open. Cancellation of the caller propagates.
    """
    if timeout <= 0:
        logger.warning("No time left to open the %s queue", purpose)
        return None
    try:
        return await asyncio.wait_for(get_arq_pool(), timeout)
    except Exception as exc:  # noqa: BLE001 - uncertain, not absent
        logger.warning("Could not open %s queue for inspection (%s)", purpose,
                       type(exc).__name__)
        return None


async def _existing_job_dispatch(pool: ArqRedis, job_id: uuid.UUID) -> DispatchOutcome:
    """A retained result or an orphan queue entry is not runnable work."""
    async def inspect() -> DispatchOutcome:
        task = ArqJob(
            str(job_id), pool, _queue_name=pool.default_queue_name,
            _deserializer=pool.job_deserializer,
        )
        status = await task.status()
        if status not in {ArqJobStatus.queued, ArqJobStatus.deferred, ArqJobStatus.in_progress}:
            return DispatchOutcome.NOT_DISPATCHED
        info = await task.info()
        if (
            info is not None and not isinstance(info, JobResult)
            and info.function == RUN_JOB_TASK and info.args == (str(job_id),)
            and not info.kwargs
        ):
            return DispatchOutcome.ALREADY_DISPATCHED
        return DispatchOutcome.NOT_DISPATCHED

    try:
        return await asyncio.wait_for(inspect(), DISPATCH_INSPECTION_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - uncertain dispatch is compensated by submission
        logger.warning("Could not confirm Job dispatch %s (%s)", job_id, type(exc).__name__)
        return DispatchOutcome.UNKNOWN


async def inspect_job_dispatches(
    job_ids: list[uuid.UUID],
) -> dict[uuid.UUID, DispatchOutcome]:
    """Whether arq still holds each Job's run_job task (same rules as dispatch).

    An unreachable queue makes every answer UNKNOWN, never NOT_DISPATCHED.
    """
    if not job_ids:
        return {}
    pool = await _open_inspection_pool(INSPECTION_POOL_TIMEOUT, "Job")
    if pool is None:
        return dict.fromkeys(job_ids, DispatchOutcome.UNKNOWN)
    try:
        return {job_id: await _existing_job_dispatch(pool, job_id) for job_id in job_ids}
    finally:
        try:
            await pool.aclose()
        except Exception as exc:  # noqa: BLE001 - inspection already finished
            logger.warning("Could not close Job queue (%s)", type(exc).__name__)


PlanAttempt = tuple[uuid.UUID, uuid.UUID]
# Plan inspection runs a few lookups at a time under one overall deadline that
# starts before the pool is opened and also covers settling cancelled lookups,
# so a slow or silent Redis holds the pass for about this long (plus closing
# the pool, itself bounded by arq's 1 s connect timeout).
PLAN_INSPECTION_CONCURRENCY = 20
PLAN_INSPECTION_BUDGET = 15.0
# Part of the budget kept for cancelling and settling unanswered lookups.
PLAN_INSPECTION_SETTLE = 1.0
_RUNNABLE = {ArqJobStatus.queued, ArqJobStatus.deferred, ArqJobStatus.in_progress}


async def _plan_dispatch(pool: ArqRedis, plan_id: uuid.UUID,
                         attempt_id: uuid.UUID) -> DispatchOutcome:
    """Whether arq can still run this attempt's run_plan task.

    arq reads the result, in-progress and queue keys in one transaction, and a
    worker finishes a task by removing it from the queue and storing its
    result atomically. A stored result (success or failure) or a missing task
    is therefore definitely not runnable; queued, deferred and in-progress
    tasks are, when they are the exact task approval enqueued.
    """
    task = ArqJob(
        plan_task_id(plan_id, attempt_id), pool, _queue_name=pool.default_queue_name,
        _deserializer=pool.job_deserializer,
    )
    if await task.status() not in _RUNNABLE:
        return DispatchOutcome.NOT_DISPATCHED
    info = await task.info()
    if (
        info is not None and not isinstance(info, JobResult)
        and info.function == RUN_PLAN_TASK
        and info.args == (str(plan_id), str(attempt_id)) and not info.kwargs
    ):
        return DispatchOutcome.ALREADY_DISPATCHED
    # A queue entry without its task definition is skipped by arq workers.
    return DispatchOutcome.NOT_DISPATCHED


async def inspect_plan_dispatches(
    pairs: list[PlanAttempt],
) -> dict[PlanAttempt, DispatchOutcome]:
    """Whether arq still holds each approved attempt's run_plan task.

    Read-only. One deadline (PLAN_INSPECTION_BUDGET) starts here, before the
    pool is opened, and covers opening it, every lookup and settling the
    cancelled ones. Anything not answered (unreachable or silent queue, a
    lookup error, the deadline) stays UNKNOWN, never NOT_DISPATCHED.
    """
    outcomes = dict.fromkeys(pairs, DispatchOutcome.UNKNOWN)
    if not pairs:
        return outcomes
    loop = asyncio.get_running_loop()
    deadline = loop.time() + PLAN_INSPECTION_BUDGET
    settle = min(PLAN_INSPECTION_SETTLE, PLAN_INSPECTION_BUDGET / 2)
    pool = await _open_inspection_pool(deadline - settle - loop.time(), "plan")
    if pool is None:
        return outcomes
    limit = asyncio.Semaphore(PLAN_INSPECTION_CONCURRENCY)

    async def inspect_one(pair: PlanAttempt) -> None:
        async with limit:
            outcomes[pair] = await _plan_dispatch(pool, *pair)

    tasks: list[asyncio.Task[None]] = []
    try:
        lookup_time = deadline - settle - loop.time()
        if lookup_time <= 0:
            logger.warning("No time left to inspect %d plan attempts", len(outcomes))
            return outcomes
        tasks = [asyncio.create_task(inspect_one(pair)) for pair in outcomes]
        done, pending = await asyncio.wait(tasks, timeout=lookup_time)
        failed = [task for task in done if task.exception() is not None]
        if failed:
            logger.warning("Plan queue inspection failed for %d attempts (%s)", len(failed),
                           type(failed[0].exception()).__name__)
        if pending:
            logger.warning("Plan queue inspection timed out for %d attempts", len(pending))
        return outcomes
    finally:
        # Also on our own cancellation (worker shutdown): no lookup outlives
        # this call. An unanswered lookup never wrote its outcome.
        unfinished = [task for task in tasks if not task.done()]
        for task in unfinished:
            task.cancel()
        if unfinished:
            _done, stuck = await asyncio.wait(
                unfinished, timeout=max(deadline - loop.time(), 0.05))
            if stuck:
                logger.warning("%d plan queue lookups did not stop in time", len(stuck))
        try:
            await pool.aclose()
        except Exception as exc:  # noqa: BLE001 - inspection already finished
            logger.warning("Could not close plan queue (%s)", type(exc).__name__)


async def enqueue_job(job_id: uuid.UUID) -> DispatchOutcome:
    try:
        pool = await get_arq_pool()
    except Exception as exc:  # noqa: BLE001 - enqueue was never attempted
        logger.warning("Could not open Job queue %s (%s)", job_id, type(exc).__name__)
        return DispatchOutcome.NOT_DISPATCHED
    try:
        # _job_id makes our Job.id double as arq's job id, so cancel_job can
        # reference the same enqueued job without tracking a second id.
        try:
            task = await pool.enqueue_job(RUN_JOB_TASK, str(job_id), _job_id=str(job_id))
        except Exception as exc:  # noqa: BLE001 - acceptance may precede acknowledgement loss
            logger.warning("Job enqueue acknowledgement failed %s (%s)", job_id, type(exc).__name__)
            return await _existing_job_dispatch(pool, job_id)
        if task is not None:
            return DispatchOutcome.ACCEPTED
        return await _existing_job_dispatch(pool, job_id)
    finally:
        try:
            await pool.aclose()
        except Exception as exc:  # noqa: BLE001 - cleanup cannot invalidate acknowledgement
            logger.warning("Could not close Job queue %s (%s)", job_id, type(exc).__name__)


async def cancel_job(job_id: uuid.UUID) -> bool:
    """Best-effort: ask arq to abort the job (queued or already running).

    Returns whether arq confirmed the abort; the DB-side status is set by
    services.jobs.cancel_job regardless, since that's the source of truth the
    API/GUI see immediately.
    """
    pool = await get_arq_pool()
    try:
        return await ArqJob(str(job_id), pool).abort(timeout=CANCEL_TIMEOUT)
    finally:
        await pool.aclose()


def plan_task_id(plan_id: uuid.UUID, attempt_id: uuid.UUID) -> str:
    """The arq job id of an approved attempt's run_plan task."""
    return f"agent-plan-{plan_id}-{attempt_id}"


async def enqueue_plan(plan_id: uuid.UUID, attempt_id: uuid.UUID) -> None:
    pool = await get_arq_pool()
    try:
        await pool.enqueue_job(
            RUN_PLAN_TASK,
            str(plan_id),
            str(attempt_id),
            _job_id=plan_task_id(plan_id, attempt_id),
        )
    finally:
        await pool.aclose()


async def enqueue_asset_probe(asset_id: uuid.UUID) -> None:
    """Queue asynchronous metadata probing for an uploaded Asset."""
    pool = await get_arq_pool()
    try:
        await pool.enqueue_job(
            RUN_ASSET_PROBE_TASK,
            str(asset_id),
            _job_id=f"asset-probe-{asset_id}",
        )
    finally:
        await pool.aclose()
