"""arq task entrypoints."""

from __future__ import annotations

import uuid

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
