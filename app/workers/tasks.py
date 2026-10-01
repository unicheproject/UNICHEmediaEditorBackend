"""arq task entrypoints."""

from __future__ import annotations

import uuid

from app.agent.executor import execute_plan
from app.core.database import async_session_factory
from app.core.logging import get_logger
from app.services.jobs import execute_job
from app.services.media_metadata import probe_asset_metadata
from app.services.temp_assets import cleanup_expired_temp_assets

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


async def run_asset_probe(ctx: dict, asset_id: str) -> str:
    """Probe an uploaded asset without coupling asset usability to probe success."""
    logger.info("Probing technical metadata for asset %s", asset_id)
    async with async_session_factory() as session:
        status = await probe_asset_metadata(session, uuid.UUID(asset_id))
    return status.value if status is not None else "skipped"


async def cleanup_temp_assets(ctx: dict) -> int:
    """Periodically expire Temp Processed Asset binaries, retaining audit rows."""
    async with async_session_factory() as session:
        deleted = await cleanup_expired_temp_assets(session)
    logger.info("Expired %s Temp Processed Asset binaries", deleted)
    return deleted
