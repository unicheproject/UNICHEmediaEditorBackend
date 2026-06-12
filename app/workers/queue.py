"""arq queue connection + enqueue helper used by the API layer.

The API enqueues the opaque `run_job` task by id — it never references a
provider or executes work in the request handler.
"""

from __future__ import annotations

import uuid

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.core.config import settings

RUN_JOB_TASK = "run_job"
RUN_PLAN_TASK = "run_plan"


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def get_arq_pool() -> ArqRedis:
    return await create_pool(redis_settings())


async def enqueue_job(job_id: uuid.UUID) -> None:
    pool = await get_arq_pool()
    try:
        await pool.enqueue_job(RUN_JOB_TASK, str(job_id))
    finally:
        await pool.aclose()


async def enqueue_plan(plan_id: uuid.UUID) -> None:
    pool = await get_arq_pool()
    try:
        await pool.enqueue_job(RUN_PLAN_TASK, str(plan_id))
    finally:
        await pool.aclose()
