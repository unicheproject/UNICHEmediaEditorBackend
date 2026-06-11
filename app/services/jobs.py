"""Job service: creation (enqueue), retrieval, and worker-side execution."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.capabilities import registry
from app.capabilities.context import HandlerResult, JobContext
from app.core.errors import AppError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.models.asset import Asset
from app.models.enums import JobStatus
from app.models.job import Job
from app.schemas.job import JobCreate
from app.services import assets as assets_svc
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
        asset = await get_asset(session, data.asset_id)
        if asset.media_type not in cap.supported_media_types:
            raise ValidationError(
                f"Capability '{data.capability_id}' does not support "
                f"{asset.media_type.value} assets"
            )

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


async def _build_job_context(
    session: AsyncSession, job: Job, work_dir: Path
) -> JobContext:
    """Resolve input assets to paths and assemble the handler context."""
    storage = get_storage()
    input_path: str | None = None
    input_asset_meta: dict = {}
    input_paths: list[str] = []
    project_id = job.project_id

    # Collect primary asset first, then any extra asset_ids (multi-input ops).
    asset_ids: list[uuid.UUID] = []
    if job.asset_id is not None:
        asset_ids.append(job.asset_id)
    for raw in (job.input or {}).get("asset_ids", []) or []:
        asset_ids.append(raw if isinstance(raw, uuid.UUID) else uuid.UUID(str(raw)))

    for idx, aid in enumerate(asset_ids):
        asset = await get_asset(session, aid)
        path = str(storage.get_path(asset.storage_path))
        input_paths.append(path)
        if idx == 0:
            input_path = path
            input_asset_meta = {
                "original_filename": asset.original_filename,
                "media_type": asset.media_type.value,
                "mime_type": asset.mime_type,
                "checksum_sha256": asset.checksum_sha256,
            }
            project_id = project_id or asset.project_id

    return JobContext(
        capability_id=job.capability_id,
        params=dict(job.input or {}),
        input_path=input_path,
        input_asset_meta=input_asset_meta,
        input_paths=input_paths,
        work_dir=work_dir,
        project_id=project_id,
        source_asset_id=job.asset_id,
    )


async def _persist_outputs(
    session: AsyncSession, job: Job, ctx: JobContext, result: HandlerResult
) -> dict:
    """Register handler output files as derived Assets; build job.output."""
    descriptors: list[dict] = []
    for out in result.outputs:
        if ctx.project_id is None:
            raise ValidationError(
                "Cannot store derived output: job has no project or input asset"
            )
        asset: Asset = await assets_svc.create_derived_asset(
            session,
            project_id=ctx.project_id,
            source_asset_id=ctx.source_asset_id,
            src_path=out.path,
            filename=out.filename,
            media_type=out.media_type,
        )
        descriptors.append(
            {
                "asset_id": str(asset.id),
                "filename": asset.original_filename,
                "media_type": asset.media_type.value,
                "size_bytes": asset.size_bytes,
                "download_path": f"/api/v1/assets/{asset.id}/download",
            }
        )

    output = dict(result.data)
    if descriptors:
        output["outputs"] = descriptors
    return output


async def execute_job(session: AsyncSession, job_id: uuid.UUID) -> Job:
    """Run a queued job to completion. Used by the worker (and tests).

    Transitions are committed as they happen so polling/SSE see live state.
    """
    job = await get_job(session, job_id)

    job.status = JobStatus.running
    job.started_at = datetime.now(UTC)
    job.progress = 10
    await session.commit()

    work_dir = Path(tempfile.mkdtemp(prefix=f"job-{job_id}-"))
    try:
        ctx = await _build_job_context(session, job, work_dir)
        handler = registry.get_handler(job.capability_id)
        job.progress = 50
        await session.commit()

        result = await handler.run(ctx)
        output = await _persist_outputs(session, job, ctx, result)

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
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    await session.refresh(job)
    return job


async def _fail(session: AsyncSession, job: Job, message: str) -> None:
    job.status = JobStatus.failed
    job.error = message
    job.finished_at = datetime.now(UTC)
    await session.commit()
