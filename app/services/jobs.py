"""Job service: creation (enqueue), retrieval, and worker-side execution."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.capabilities import registry
from app.core.errors import AppError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.models.enums import JobStatus
from app.models.job import Job
from app.providers.base import InferenceRequest
from app.schemas.job import JobCreate
from app.services.assets import get_asset
from app.services.projects import get_project
from app.services.storage import get_storage

logger = get_logger(__name__)


async def create_job(session: AsyncSession, data: JobCreate) -> Job:
    """Validate references and persist a queued job. Does NOT execute work."""
    cap = registry.get(data.capability_id)  # raises NotFoundError if unknown
    if not cap.enabled:
        raise ValidationError(f"Capability '{data.capability_id}' is disabled")

    if data.project_id is not None:
        await get_project(session, data.project_id)
    if data.asset_id is not None:
        await get_asset(session, data.asset_id)

    job = Job(
        project_id=data.project_id,
        asset_id=data.asset_id,
        capability_id=data.capability_id,
        status=JobStatus.queued,
        input=data.input,
        progress=0,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def get_job(session: AsyncSession, job_id: uuid.UUID) -> Job:
    job = await session.get(Job, job_id)
    if job is None:
        raise NotFoundError(f"Job '{job_id}' not found")
    return job


async def list_jobs_for_project(
    session: AsyncSession, project_id: uuid.UUID
) -> list[Job]:
    await get_project(session, project_id)
    result = await session.execute(
        select(Job)
        .where(Job.project_id == project_id)
        .order_by(Job.created_at.desc())
    )
    return list(result.scalars().all())


async def _build_inference_request(
    session: AsyncSession, job: Job
) -> InferenceRequest:
    file_path: str | None = None
    asset_meta: dict = {}
    if job.asset_id is not None:
        asset = await get_asset(session, job.asset_id)
        file_path = str(get_storage().get_path(asset.storage_path))
        asset_meta = {
            "original_filename": asset.original_filename,
            "media_type": asset.media_type.value,
            "mime_type": asset.mime_type,
            "checksum_sha256": asset.checksum_sha256,
        }
    payload = dict(job.input or {})
    return InferenceRequest(
        capability_id=job.capability_id,
        payload=payload,
        file_path=file_path,
        asset_meta=asset_meta,
    )


async def execute_job(session: AsyncSession, job_id: uuid.UUID) -> Job:
    """Run a queued job to completion. Used by the worker (and tests).

    Transitions are committed as they happen so polling/SSE see live state.
    """
    job = await get_job(session, job_id)

    job.status = JobStatus.running
    job.started_at = datetime.now(UTC)
    job.progress = 10
    await session.commit()

    try:
        request = await _build_inference_request(session, job)
        handler = registry.get_handler(job.capability_id)
        job.progress = 50
        await session.commit()

        output = await handler.run(request)

        job.output = output
        job.status = JobStatus.succeeded
        job.progress = 100
        job.finished_at = datetime.now(UTC)
        await session.commit()
    except AppError as exc:
        await _fail(session, job, exc.message)
    except Exception as exc:  # noqa: BLE001 - record any failure on the job
        logger.exception("Job %s failed", job_id)
        await _fail(session, job, str(exc))

    await session.refresh(job)
    return job


async def _fail(session: AsyncSession, job: Job, message: str) -> None:
    job.status = JobStatus.failed
    job.error = message
    job.finished_at = datetime.now(UTC)
    await session.commit()
