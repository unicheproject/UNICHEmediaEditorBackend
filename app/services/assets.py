"""Asset service: upload validation, persistence, retrieval, soft delete."""

from __future__ import annotations

import mimetypes
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import String, cast, func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import (
    AppError,
    GoneError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from app.core.logging import get_logger
from app.models.asset import Asset
from app.models.enums import AiProvenance, MediaType, ProvenanceKind, TechnicalMetadataStatus
from app.schemas.asset import (
    AssetBulkDeleteItem,
    AssetBulkDeleteRequest,
    AssetBulkDeleteResponse,
    AssetSearchQuery,
    AssetUpdate,
)
from app.services.media_metadata import apply_metadata, probe_media_safely
from app.services.projects import get_project
from app.services.storage import get_storage, safe_filename
from app.services.waveforms import enrich_asset_waveform, initial_waveform_state

logger = get_logger(__name__)


def _resolve_media_type(extension: str) -> MediaType:
    ext = extension.lower()
    if ext in settings.allowed_image_extensions:
        return MediaType.image
    if ext in settings.allowed_audio_extensions:
        return MediaType.audio
    if ext in settings.allowed_video_extensions:
        return MediaType.video
    if ext in settings.allowed_subtitle_extensions:
        return MediaType.subtitle
    return MediaType.unknown


def _allowed_extensions() -> set[str]:
    return (
        settings.allowed_image_extensions
        | settings.allowed_audio_extensions
        | settings.allowed_video_extensions
        | settings.allowed_subtitle_extensions
    )


async def create_asset(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    original_filename: str,
    src: BinaryIO,
    content_type: str | None,
    derived_from_job_id: uuid.UUID | None = None,
) -> Asset:
    # Project must exist (and not be soft-deleted).
    await get_project(session, project_id)

    extension = Path(original_filename).suffix.lstrip(".").lower()
    if extension not in _allowed_extensions():
        raise UnsupportedMediaTypeError(
            f"Unsupported file extension '.{extension}'. Allowed: "
            f"{', '.join(sorted(_allowed_extensions()))}"
        )

    media_type = _resolve_media_type(extension)
    source_asset_id = None
    ai_provenance = AiProvenance.none
    if derived_from_job_id is not None:
        from app.services.provenance import (
            contributing_asset_ids,
            materialization_job,
            output_ai_provenance,
        )

        job = await materialization_job(
            session, authorized_project_id=project_id, job_id=derived_from_job_id,
            media_type=media_type,
        )
        source_asset_id = job.asset_id
        ai_provenance = await output_ai_provenance(
            session, project_id=project_id, capability_ids=[job.capability_id],
            asset_ids=contributing_asset_ids(job),
        )
    stored_name = safe_filename(original_filename)
    asset_id = uuid.uuid4()
    storage = get_storage()

    storage_path, size_bytes = storage.save_upload(
        project_id, asset_id, stored_name, src
    )

    if size_bytes > settings.max_upload_size_bytes:
        # Roll back the just-written file before failing.
        try:
            storage.delete_asset(project_id, asset_id)
        except OSError as exc:
            logger.warning(
                "Could not remove rejected upload %s (%s)", asset_id, type(exc).__name__
            )
        raise PayloadTooLargeError(
            f"File exceeds maximum upload size of {settings.max_upload_size_mb} MB"
        )

    checksum = storage.calculate_checksum(storage_path)
    mime_type = content_type or mimetypes.guess_type(original_filename)[0] or (
        "application/octet-stream"
    )

    waveform_status, waveform_reason = initial_waveform_state(media_type)
    asset = Asset(
        id=asset_id,
        project_id=project_id,
        filename=stored_name,
        original_filename=original_filename,
        media_type=media_type,
        mime_type=mime_type,
        extension=extension,
        size_bytes=size_bytes,
        storage_path=storage_path,
        checksum_sha256=checksum,
        source_asset_id=source_asset_id,
        provenance_kind=(
            ProvenanceKind.derived.value if derived_from_job_id else ProvenanceKind.original.value
        ),
        ai_provenance=ai_provenance.value,
        source_job_id=derived_from_job_id,
        technical_metadata_status=TechnicalMetadataStatus.pending.value,
        waveform_status=waveform_status.value,
        waveform_reason=waveform_reason.value if waveform_reason else None,
    )
    session.add(asset)
    await session.commit()
    await session.refresh(asset)
    return asset


async def list_assets(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    include_intermediate: bool = False,
) -> list[Asset]:
    await get_project(session, project_id)
    stmt = select(Asset).where(
        Asset.project_id == project_id, Asset.deleted_at.is_(None)
    )
    if not include_intermediate:
        stmt = stmt.where(Asset.is_intermediate.is_(False))
    result = await session.execute(stmt.order_by(Asset.created_at.desc()))
    return list(result.scalars().all())


async def search_assets(
    session: AsyncSession, project_id: uuid.UUID, query: AssetSearchQuery,
) -> tuple[list[Asset], int]:
    """Read a filtered page of active Assets within an already authorized project."""
    await get_project(session, project_id)
    stmt = select(Asset).where(
        Asset.project_id == project_id, Asset.deleted_at.is_(None)
    )
    if not query.include_intermediate:
        stmt = stmt.where(Asset.is_intermediate.is_(False))
    if query.media_type is not None:
        stmt = stmt.where(Asset.media_type == query.media_type)
    if query.q:
        stmt = stmt.where(Asset.original_filename.icontains(query.q, autoescape=True))

    total = await session.scalar(select(func.count()).select_from(stmt.subquery()))
    # Cast native PostgreSQL enums so type ordering matches the public alphabetical contract.
    column = {
        "created_at": Asset.created_at,
        "name": func.lower(Asset.original_filename),
        "size": Asset.size_bytes,
        "type": cast(Asset.media_type, String),
    }[query.sort_by]
    ordering = (
        (column.asc(), Asset.id.asc()) if query.sort_order == "asc"
        else (column.desc(), Asset.id.desc())
    )
    result = await session.execute(
        stmt.order_by(*ordering).limit(query.limit).offset(query.offset)
    )
    return list(result.scalars().all()), int(total or 0)


async def get_asset(session: AsyncSession, asset_id: uuid.UUID) -> Asset:
    asset = await session.get(Asset, asset_id)
    if asset is None or asset.deleted_at is not None:
        raise NotFoundError(f"Asset '{asset_id}' not found")
    return asset


async def soft_delete_asset(session: AsyncSession, asset_id: uuid.UUID) -> None:
    asset = await get_asset(session, asset_id)
    asset.deleted_at = datetime.now(UTC)
    await session.commit()


async def bulk_soft_delete_assets(
    session: AsyncSession, project_id: uuid.UUID, data: AssetBulkDeleteRequest,
) -> AssetBulkDeleteResponse:
    """Soft-delete only active, project-owned matches in a single transaction.

    The caller must authorize the named project before entering this service.
    RETURNING reports actual updates, including under concurrent deletion;
    missing, deleted and foreign-project IDs deliberately remain indistinguishable.
    """
    await get_project(session, project_id)
    try:
        result = await session.execute(
            update(Asset)
            .where(
                Asset.project_id == project_id,
                Asset.id.in_(data.asset_ids),
                Asset.deleted_at.is_(None),
            )
            .values(deleted_at=datetime.now(UTC))
            .returning(Asset.id)
        )
        deleted_ids = set(result.scalars().all())
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        logger.error("Asset batch deletion failed (%s)", type(exc).__name__)
        raise AppError("Assets could not be deleted.") from exc

    return AssetBulkDeleteResponse(results=[
        AssetBulkDeleteItem(
            asset_id=asset_id,
            status="deleted" if asset_id in deleted_ids else "not_found",
        )
        for asset_id in data.asset_ids
    ])


async def update_asset(
    session: AsyncSession, asset_id: uuid.UUID, data: AssetUpdate
) -> Asset:
    """Rename an asset: local metadata only, no physical file/storage_path change."""
    asset = await get_asset(session, asset_id)
    fields = data.model_dump(exclude_unset=True)
    if "original_filename" in fields:
        asset.original_filename = fields["original_filename"]
    await session.commit()
    await session.refresh(asset)
    return asset


async def create_derived_asset(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    source_asset_id: uuid.UUID | None,
    src_path: Path,
    filename: str,
    media_type: MediaType,
    is_intermediate: bool = False,
    ai_provenance: AiProvenance = AiProvenance.unknown,
    source_job_id: uuid.UUID | None = None,
) -> Asset:
    """Register a tool-produced file as a new (derived) Asset.

    Trusted output: skips the upload-size cap and allowed-extension gate that
    apply to user uploads. Sets the input anchor for chaining. Callers without
    recorded execution evidence retain unknown AI provenance.
    """
    asset = await prepare_derived_asset(
        project_id=project_id, source_asset_id=source_asset_id,
        src_path=src_path, filename=filename, media_type=media_type,
        is_intermediate=is_intermediate,
        ai_provenance=ai_provenance, source_job_id=source_job_id,
    )
    session.add(asset)
    await session.commit()
    await session.refresh(asset)
    if media_type in (MediaType.audio, MediaType.video):
        await enrich_asset_waveform(session, asset.id)
        await session.refresh(asset)
    return asset


def discard_prepared_asset(project_id: uuid.UUID, asset_id: uuid.UUID) -> None:
    """Best-effort cleanup of an unpublished, freshly allocated object."""
    try:
        get_storage().delete_asset(project_id, asset_id)
    except Exception as exc:  # noqa: BLE001 - cleanup must preserve the original failure
        logger.warning("Could not remove unpublished asset %s (%s)", asset_id, type(exc).__name__)


async def prepare_derived_asset(
    *,
    project_id: uuid.UUID,
    source_asset_id: uuid.UUID | None,
    src_path: Path,
    filename: str,
    media_type: MediaType,
    is_intermediate: bool = False,
    ai_provenance: AiProvenance = AiProvenance.unknown,
    source_job_id: uuid.UUID | None = None,
) -> Asset:
    """Prepare storage and a transient Asset; the caller owns DB publication.

    No session writes or preview enrichment occur here. A failed preparation
    removes its own object, including a partially copied file.
    """
    stored_name = safe_filename(filename)
    extension = Path(filename).suffix.lstrip(".").lower()
    asset_id = uuid.uuid4()
    storage = get_storage()

    try:
        with src_path.open("rb") as src:
            storage_path, size_bytes = storage.save_upload(
                project_id, asset_id, stored_name, src
            )
        checksum = storage.calculate_checksum(storage_path)
        metadata = await probe_media_safely(src_path, media_type)
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        persisted_at = datetime.now(UTC)
        waveform_status, waveform_reason = initial_waveform_state(media_type)
        asset = Asset(
            id=asset_id,
            project_id=project_id,
            filename=stored_name,
            original_filename=filename,
            media_type=media_type,
            mime_type=mime_type,
            extension=extension,
            size_bytes=size_bytes,
            storage_path=storage_path,
            checksum_sha256=checksum,
            source_asset_id=source_asset_id,
            provenance_kind=ProvenanceKind.derived.value,
            ai_provenance=ai_provenance.value,
            source_job_id=source_job_id,
            is_intermediate=is_intermediate,
            temp_expires_at=(
                persisted_at + timedelta(hours=72) if is_intermediate else None
            ),
            technical_metadata_status=metadata.status.value,
            waveform_status=waveform_status.value,
            waveform_reason=waveform_reason.value if waveform_reason else None,
        )
        apply_metadata(asset, metadata)
        return asset
    except BaseException:
        discard_prepared_asset(project_id, asset_id)
        raise


def ensure_asset_binary_available(asset: Asset) -> Path:
    """Return a readable local media path or a controlled lifecycle error."""
    if asset.physical_expired_at is not None:
        raise GoneError("This Asset's temporary media has expired")
    path = get_storage().get_path(asset.storage_path)
    if not path.is_file():
        raise NotFoundError("Asset media is not available")
    return path
