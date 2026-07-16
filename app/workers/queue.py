"""arq queue connection + enqueue helper used by the API layer.

The API enqueues the opaque `run_job` task by id — it never references a
provider or executes work in the request handler.
"""

from __future__ import annotations

import uuid

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from arq.jobs import Job as ArqJob

from app.core.config import settings

RUN_JOB_TASK = "run_job"
RUN_PLAN_TASK = "run_plan"

# How long cancel_job() waits to hear back from arq before giving up (the DB
# is still updated immediately by services.jobs.cancel_job regardless).
CANCEL_TIMEOUT = 5.0


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def get_arq_pool() -> ArqRedis:
    return await create_pool(redis_settings())


async def enqueue_job(job_id: uuid.UUID) -> None:
    pool = await get_arq_pool()
    try:
        # _job_id makes our Job.id double as arq's job id, so cancel_job can
        # reference the same enqueued job without tracking a second id.
        await pool.enqueue_job(RUN_JOB_TASK, str(job_id), _job_id=str(job_id))
    finally:
        await pool.aclose()


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


async def enqueue_plan(plan_id: uuid.UUID) -> None:
    pool = await get_arq_pool()
    try:
        await pool.enqueue_job(RUN_PLAN_TASK, str(plan_id))
    finally:
        await pool.aclose()
