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

# How long cancel_job() waits to hear back from arq before giving up (the DB
# is still updated immediately by services.jobs.cancel_job regardless).
CANCEL_TIMEOUT = 5.0
DISPATCH_INSPECTION_TIMEOUT = 5.0


class DispatchOutcome(Enum):
    ACCEPTED = "accepted"
    ALREADY_DISPATCHED = "already_dispatched"
    NOT_DISPATCHED = "not_dispatched"
    UNKNOWN = "unknown"


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def get_arq_pool() -> ArqRedis:
    return await create_pool(redis_settings())


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


async def enqueue_plan(plan_id: uuid.UUID, attempt_id: uuid.UUID) -> None:
    pool = await get_arq_pool()
    try:
        await pool.enqueue_job(
            RUN_PLAN_TASK,
            str(plan_id),
            str(attempt_id),
            _job_id=f"agent-plan-{plan_id}-{attempt_id}",
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
