"""arq task entrypoints."""

from __future__ import annotations

import uuid

from app.agent.executor import execute_plan
from app.core.database import async_session_factory
from app.core.logging import get_logger
from app.services.jobs import execute_job

logger = get_logger(__name__)


async def run_job(ctx: dict, job_id: str) -> str:
    """Execute a queued job. `ctx` is the arq worker context (unused here)."""
    logger.info("Running job %s", job_id)
    async with async_session_factory() as session:
        job = await execute_job(session, uuid.UUID(job_id))
    logger.info("Job %s finished with status=%s", job_id, job.status.value)
    return job.status.value


async def run_plan(ctx: dict, plan_id: str) -> str:
    """Execute an approved agent plan as a chained job sequence."""
    logger.info("Running plan %s", plan_id)
    async with async_session_factory() as session:
        plan = await execute_plan(session, uuid.UUID(plan_id))
    logger.info("Plan %s finished with status=%s", plan_id, plan.status.value)
    return plan.status.value
