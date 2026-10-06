"""Job service: creation (enqueue), retrieval, and worker-side execution."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from sqlalchemy import func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.capabilities import registry
from app.capabilities.applicability import validate_job_applicability
from app.capabilities.context import HandlerResult, JobContext
from app.capabilities.readiness import require_executable, require_worker_readiness
from app.capabilities.validation import (
    CapabilityInputValidationError,
    iter_media_asset_references,
    validate_capability_input,
)
from app.core.errors import AppError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.models.asset import Asset
from app.models.enums import JobStatus, MediaType
from app.models.job import Job
from app.schemas.capability import CapabilityMetadata
from app.schemas.job import JobCreate
from app.services import assets as assets_svc
from app.services.assets import ensure_asset_binary_available, get_asset
from app.services.projects import get_project
from app.services.storage import get_storage
from app.services.waveforms import enrich_asset_waveform
from app.workers.queue import DispatchOutcome

logger = get_logger(__name__)
DISPATCH_FAILURE_MESSAGE = "Job could not be dispatched for execution."

# None remains an acknowledgement for existing injected/eager enqueuers;
# the production queue wrapper always returns an explicit DispatchOutcome.
JobEnqueuer = Callable[[uuid.UUID], Awaitable[DispatchOutcome | None]]


async def create_job(session: AsyncSession, data: JobCreate) -> Job:
    """Validate references and persist a queued job. Does NOT execute work."""
    cap = registry.get(data.capability_id)  # raises NotFoundError if unknown
    require_executable(cap)
    validate_job_input(cap, data.input)
    validate_job_applicability(
        cap,
        asset_id=data.asset_id,
        requested_project_id=data.project_id,
    )

    project_id = await resolve_job_project(
        session,
        project_id=data.project_id,
        asset_id=data.asset_id,
        job_input=data.input,
        lock_assets=True,
    )
    validate_job_applicability(
        cap,
        asset_id=data.asset_id,
        requested_project_id=data.project_id,
        resolved_project_id=project_id,
        project_resolution_complete=True,
    )
    if project_id is not None:
        await get_project(session, project_id)
    stored_input = dict(data.input)
    if data.capability_id == "project.export":
        if project_id is None:
            raise ValidationError("project.export requires project context")
        from app.services.project_exports import create_export_execution_snapshot

        stored_input["execution_snapshot"] = await create_export_execution_snapshot(
            session, project_id=project_id, request=dict(data.input)
        )
    if data.asset_id is not None:
        asset = await get_asset(session, data.asset_id)
        if asset.media_type not in cap.supported_media_types:
            raise ValidationError(
                f"Capability '{data.capability_id}' does not support "
                f"{asset.media_type.value} assets"
            )
    await _validate_referenced_asset_media_types(session, cap, data.input)

    job = Job(
        project_id=project_id,
        asset_id=data.asset_id,
        capability_id=data.capability_id,
        status=JobStatus.queued,
        input=stored_input,
        progress=0,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def fail_queued_dispatch(session: AsyncSession, job_id: uuid.UUID) -> bool:
    """Compensate caught dispatch failure only if no worker/cancellation won."""
    await session.rollback()
    try:
        changed = await session.execute(
            update(Job).where(Job.id == job_id, Job.status == JobStatus.queued)
            .values(status=JobStatus.failed, error=DISPATCH_FAILURE_MESSAGE,
                    finished_at=datetime.now(UTC))
            .execution_options(synchronize_session=False)
        )
        await session.commit()
    except Exception:  # noqa: BLE001 - leave a usable session and sanitize at submission
        await session.rollback()
        raise
    return cast(CursorResult, changed).rowcount == 1


async def dispatch_job(
    session: AsyncSession, job_id: uuid.UUID, enqueuer: JobEnqueuer,
) -> Job:
    """Dispatch an already committed Job, or truthfully terminate queued work."""
    # End the creation refresh's read transaction before network I/O and races.
    await session.commit()
    try:
        outcome = await enqueuer(job_id)
    except Exception as exc:  # noqa: BLE001 - injected transports may also raise
        logger.warning("Job dispatch failed %s (%s)", job_id, type(exc).__name__)
        outcome = DispatchOutcome.UNKNOWN
    if outcome in {DispatchOutcome.NOT_DISPATCHED, DispatchOutcome.UNKNOWN}:
        try:
            compensated = await fail_queued_dispatch(session, job_id)
            job = await get_job(session, job_id)
            await session.refresh(job)
        except Exception as exc:  # noqa: BLE001 - never expose database/transport diagnostics
            logger.warning("Job dispatch compensation failed %s (%s)", job_id, type(exc).__name__)
            raise AppError(DISPATCH_FAILURE_MESSAGE) from None
        if compensated:
            raise AppError(DISPATCH_FAILURE_MESSAGE)
        return job
    job = await get_job(session, job_id)
    await session.refresh(job)
    return job


def validate_job_input(capability: CapabilityMetadata, job_input: dict) -> None:
    """Map canonical schema failures to the application's 422 error contract."""
    try:
        validate_capability_input(capability, job_input)
    except CapabilityInputValidationError as exc:
        raise ValidationError(f"Invalid capability input: {exc}") from exc


async def _validate_referenced_asset_media_types(
    session: AsyncSession, capability: CapabilityMetadata, job_input: dict
) -> None:
    """Enforce x-media-asset declarations after project isolation checks."""
    for reference in iter_media_asset_references(capability, job_input):
        asset = await get_asset(session, uuid.UUID(reference.asset_id))
        if reference.media_types and asset.media_type.value not in reference.media_types:
            raise ValidationError(
                f"Parameter '{reference.parameter}' requires "
                f"{', '.join(reference.media_types)} media, but asset "
                f"'{reference.asset_id}' is {asset.media_type.value}"
            )


def _literal_asset_references(
    asset_id: uuid.UUID | None, job_input: dict | None
) -> list[uuid.UUID]:
    """Return every literal asset UUID represented by the current job contract."""
    references: list[uuid.UUID] = []
    if asset_id is not None:
        references.append(asset_id)

    values = job_input or {}
    for raw in values.get("asset_ids", []) or []:
        references.append(raw if isinstance(raw, uuid.UUID) else uuid.UUID(str(raw)))
    for key, value in values.items():
        if key.endswith("_asset_id") and value:
            references.append(
                value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
            )
    return references


async def resolve_job_project(
    session: AsyncSession,
    *,
    project_id: uuid.UUID | None,
    asset_id: uuid.UUID | None,
    job_input: dict | None,
    lock_assets: bool = False,
) -> uuid.UUID | None:
    """Resolve and enforce the single-project boundary for all job assets."""
    resolved_project_id = project_id
    for reference_id in _literal_asset_references(asset_id, job_input):
        if lock_assets:
            result = await session.execute(
                select(Asset)
                .where(Asset.id == reference_id, Asset.deleted_at.is_(None))
                .with_for_update()
            )
            asset = result.scalar_one_or_none()
            if asset is None:
                raise NotFoundError(f"Asset '{reference_id}' not found")
        else:
            asset = await get_asset(session, reference_id)
        if resolved_project_id is None:
            resolved_project_id = asset.project_id
        elif asset.project_id != resolved_project_id:
            raise ValidationError(
                f"Asset '{reference_id}' is not in the job's project"
            )
        ensure_asset_binary_available(asset)
    return resolved_project_id


async def infer_job_project(
    session: AsyncSession,
    *,
    asset_id: uuid.UUID | None,
    job_input: dict | None,
) -> uuid.UUID | None:
    """Infer scope from only the first asset reference when no project is given.

    The API authorizes this candidate project before resolving the remaining
    user-supplied references. Full consistency validation remains in
    ``resolve_job_project`` and is also performed by service/worker callers.
    """
    references = _literal_asset_references(asset_id, job_input)
    if not references:
        return None
    asset = await get_asset(session, references[0])
    return asset.project_id


async def get_job(session: AsyncSession, job_id: uuid.UUID) -> Job:
    job = await session.get(Job, job_id)
    if job is None:
        raise NotFoundError(f"Job '{job_id}' not found")
    return job


_CANCELLABLE = {JobStatus.queued, JobStatus.running}


async def cancel_job(session: AsyncSession, job_id: uuid.UUID) -> Job:
    """Mark a queued/running job cancelled. The route also signals arq (best-
    effort) to actually stop a running job's subprocess; this DB write is the
    immediately-consistent state the API/GUI see regardless of that outcome.
    """
    changed = await session.execute(
        update(Job).where(Job.id == job_id, Job.status.in_(_CANCELLABLE))
        .values(status=JobStatus.cancelled, error="Cancelled by user",
                finished_at=datetime.now(UTC))
        .execution_options(synchronize_session=False)
    )
    await session.commit()
    job = await get_job(session, job_id)
    await session.refresh(job)
    if cast(CursorResult, changed).rowcount != 1:
        raise ValidationError(
            f"Job '{job_id}' is already {job.status.value}; cannot cancel"
        )
    return job


async def claim_job(session: AsyncSession, job_id: uuid.UUID) -> bool:
    """One attempt per Job: only the atomic queued -> running winner executes."""
    changed = await session.execute(
        update(Job).where(Job.id == job_id, Job.status == JobStatus.queued)
        .values(status=JobStatus.running, started_at=datetime.now(UTC), progress=10)
        .execution_options(synchronize_session=False)
    )
    await session.commit()
    return cast(CursorResult, changed).rowcount == 1


async def finish_running_job(
    session: AsyncSession, job_id: uuid.UUID, status: JobStatus, message: str,
) -> None:
    """Failure/abort cannot overwrite a cancellation or successful publication."""
    if status not in {JobStatus.failed, JobStatus.cancelled}:
        raise ValueError("Only failure/abort may finish a running Job here")
    await session.rollback()
    await session.execute(
        update(Job).where(Job.id == job_id, Job.status == JobStatus.running)
        .values(status=status, error=message, finished_at=datetime.now(UTC))
        .execution_options(synchronize_session=False)
    )
    await session.commit()


async def list_jobs_for_project(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    limit: int,
    offset: int,
    status: JobStatus | None = None,
) -> tuple[list[Job], int]:
    """Return one page of a project's jobs (newest first) plus the total count."""
    await get_project(session, project_id)
    base = select(Job).where(Job.project_id == project_id)
    if status is not None:
        base = base.where(Job.status == status)
    total = await session.scalar(
        select(func.count()).select_from(base.subquery())
    )
    result = await session.execute(
        base.order_by(Job.created_at.desc()).limit(limit).offset(offset)
    )
    return list(result.scalars().all()), int(total or 0)


async def _build_job_context(
    session: AsyncSession, job: Job, work_dir: Path
) -> JobContext:
    """Resolve input assets to paths and assemble the handler context."""
    capability = registry.get(job.capability_id)
    job_input = dict(job.input or {})
    request_input = (
        {key: value for key, value in job_input.items() if key != "execution_snapshot"}
        if job.capability_id == "project.export"
        else job_input
    )
    validate_job_input(capability, request_input)
    project_id = await resolve_job_project(
        session,
        project_id=job.project_id,
        asset_id=job.asset_id,
        job_input=request_input,
    )
    await _validate_referenced_asset_media_types(session, capability, request_input)

    if job.capability_id == "project.export":
        if project_id is None or job.project_id != project_id or job.asset_id is not None:
            raise ValidationError("Project export Job scope is invalid")
        from app.services.project_exports import resolve_project_export_execution

        execution = await resolve_project_export_execution(
            session, project_id=project_id, job_input=job_input
        )
        video_ids = list(execution.compiled.request.video_asset_ids)
        video_paths = list(execution.sources.video_paths)
        first_path = video_paths[0] if video_paths else None
        audio_path = execution.sources.audio_path
        return JobContext(
            capability_id=job.capability_id,
            params=request_input,
            input_path=first_path,
            input_paths=video_paths,
            input_asset_ids=video_ids,
            named_input_paths=(
                {"audio_asset_id": audio_path} if audio_path is not None else {}
            ),
            work_dir=work_dir,
            project_id=project_id,
            source_asset_id=video_ids[0] if video_ids else None,
            project_export=execution,
        )
    storage = get_storage()
    input_path: str | None = None
    input_asset_meta: dict = {}
    input_paths: list[str] = []

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

    # Resolve named secondary inputs: any param key ending in "_asset_id"
    # (e.g. subtitle_asset_id, music_asset_id, audio_asset_id) -> file path.
    named_input_paths: dict[str, str] = {}
    for key, value in (job.input or {}).items():
        if key.endswith("_asset_id") and value:
            named_asset = await get_asset(session, uuid.UUID(str(value)))
            named_input_paths[key] = str(storage.get_path(named_asset.storage_path))
            project_id = project_id or named_asset.project_id

    # Provenance anchor: the primary asset, else the first input segment.
    source_asset_id = asset_ids[0] if asset_ids else None

    return JobContext(
        capability_id=job.capability_id,
        params=dict(job.input or {}),
        input_path=input_path,
        input_asset_meta=input_asset_meta,
        input_paths=input_paths,
        input_asset_ids=asset_ids,
        named_input_paths=named_input_paths,
        work_dir=work_dir,
        project_id=project_id,
        source_asset_id=source_asset_id,
    )


async def _persist_outputs(
    session: AsyncSession,
    job: Job,
    ctx: JobContext,
    result: HandlerResult,
    *,
    is_intermediate: bool = False,
) -> bool:
    """Prepare objects, then atomically publish ALL Assets and Job success.

    The conditional UPDATE serializes against cancellation on both PostgreSQL
    and SQLite. Its write lock lasts only through the publication commit, never
    media processing. No row is visible if cancellation wins or commit fails.
    """
    job_id = job.id
    prepared: list[Asset] = []
    prepared_keys: list[tuple[uuid.UUID, uuid.UUID]] = []
    published = False
    # End any context-resolution read transaction before media probing/copying.
    await session.commit()
    try:
        for out in result.outputs:
            if ctx.project_id is None:
                raise ValidationError(
                    "Cannot store derived output: job has no project or input asset"
                )
            asset = await assets_svc.prepare_derived_asset(
                project_id=ctx.project_id, source_asset_id=ctx.source_asset_id,
                src_path=out.path, filename=out.filename, media_type=out.media_type,
                is_intermediate=is_intermediate,
            )
            prepared.append(asset)
            prepared_keys.append((asset.project_id, asset.id))
        output = dict(result.data)
        if prepared:
            output["outputs"] = [{
                "asset_id": str(asset.id), "filename": asset.original_filename,
                "media_type": asset.media_type.value, "size_bytes": asset.size_bytes,
                "download_path": f"/api/v1/assets/{asset.id}/download",
            } for asset in prepared]
        changed = await session.execute(
            update(Job).where(Job.id == job_id, Job.status == JobStatus.running)
            .values(status=JobStatus.succeeded, output=output, progress=100,
                    error=None, finished_at=datetime.now(UTC))
            .execution_options(synchronize_session=False)
        )
        if cast(CursorResult, changed).rowcount != 1:
            await session.rollback()
            return False
        session.add_all(prepared)
        # Resolve cooperative cancellation during commit before deciding whether
        # these objects are published; never delete an acknowledged commit.
        commit = asyncio.create_task(session.commit())
        try:
            await asyncio.shield(commit)
        except asyncio.CancelledError:
            await commit
            published = True
            raise
        published = True
    except BaseException:
        await session.rollback()
        raise
    finally:
        if not published:
            await _discard_unpublished_outputs(session, prepared_keys)

    # Preview enrichment has independent best-effort transactions only AFTER
    # authoritative publication. Its failure cannot revoke Job success.
    enrichment_ids = [
        asset.id for asset in prepared if asset.media_type in (MediaType.audio, MediaType.video)
    ]
    for asset_id in enrichment_ids:
        await enrich_asset_waveform(session, asset_id)
    return True


async def _discard_unpublished_outputs(
    session: AsyncSession, keys: list[tuple[uuid.UUID, uuid.UUID]],
) -> None:
    """Never remove committed objects if a commit acknowledgement was lost.

    If the DB cannot confirm ownership, retain objects rather than risking
    published media. Abrupt death/DB outage can leave unreferenced storage;
    reconciliation is deliberately outside the one-attempt execution contract.
    """
    if not keys:
        return
    try:
        async with AsyncSession(bind=session.bind) as checking:
            committed = set(await checking.scalars(
                select(Asset.id).where(Asset.id.in_([asset_id for _, asset_id in keys]))
            ))
    except Exception as exc:  # noqa: BLE001 - do not delete without authoritative evidence
        logger.warning("Cannot verify unpublished output cleanup (%s)", type(exc).__name__)
        return
    for project_id, asset_id in keys:
        if asset_id not in committed:
            assets_svc.discard_prepared_asset(project_id, asset_id)


async def execute_job(
    session: AsyncSession,
    job_id: uuid.UUID,
    *,
    is_intermediate_output: bool = False,
) -> Job:
    """Run a queued job to completion. Used by the worker (and tests).

    Transitions are committed as they happen so polling/SSE see live state.
    """
    claimed = await claim_job(session, job_id)
    job = await get_job(session, job_id)
    await session.refresh(job)
    if not claimed:
        return job

    work_dir = Path(tempfile.mkdtemp(prefix=f"job-{job_id}-"))
    try:
        await require_worker_readiness(registry.get(job.capability_id))
        ctx = await _build_job_context(session, job, work_dir)
        handler = registry.get_handler(job.capability_id)
        changed = await session.execute(
            update(Job).where(Job.id == job_id, Job.status == JobStatus.running)
            .values(progress=50).execution_options(synchronize_session=False)
        )
        await session.commit()
        if cast(CursorResult, changed).rowcount != 1:
            await session.refresh(job)
            return job

        result = await handler.run(ctx)
        await _persist_outputs(
            session,
            job,
            ctx,
            result,
            is_intermediate=is_intermediate_output,
        )

    except asyncio.CancelledError:
        # Cooperative ARQ aborts must not strand the physical Job as running.
        # The runner handles killing an active subprocess before this point.
        await finish_running_job(
            session, job_id, JobStatus.cancelled,
            "Worker execution was cooperatively cancelled.",
        )
        raise
    except AppError as exc:
        await finish_running_job(session, job_id, JobStatus.failed, exc.message)
    except Exception as exc:  # noqa: BLE001 - record any failure on the job
        logger.exception("Job %s failed", job_id)
        await finish_running_job(session, job_id, JobStatus.failed, str(exc))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    await session.refresh(job)
    return job
