"""Asset service: upload validation, persistence, retrieval, soft delete."""

from __future__ import annotations

import mimetypes
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import (
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from app.models.asset import Asset
from app.models.enums import MediaType
from app.schemas.asset import AssetUpdate
from app.services.projects import get_project
from app.services.storage import get_storage, safe_filename


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
    stored_name = safe_filename(original_filename)
    asset_id = uuid.uuid4()
    storage = get_storage()

    storage_path, size_bytes = storage.save_upload(
        project_id, asset_id, stored_name, src
    )

    if size_bytes > settings.max_upload_size_bytes:
        # Roll back the just-written file before failing.
        storage.delete_asset(project_id, asset_id)
        raise PayloadTooLargeError(
            f"File exceeds maximum upload size of {settings.max_upload_size_mb} MB"
        )

    checksum = storage.calculate_checksum(storage_path)
    mime_type = content_type or mimetypes.guess_type(original_filename)[0] or (
        "application/octet-stream"
    )

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
    )
    session.add(asset)
    await session.commit()
    await session.refresh(asset)
    return asset


async def list_assets(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    include_intermediate: bool = True,
) -> list[Asset]:
    await get_project(session, project_id)
    stmt = select(Asset).where(
        Asset.project_id == project_id, Asset.deleted_at.is_(None)
    )
    if not include_intermediate:
        stmt = stmt.where(Asset.is_intermediate.is_(False))
    result = await session.execute(stmt.order_by(Asset.created_at.desc()))
    return list(result.scalars().all())


async def get_asset(session: AsyncSession, asset_id: uuid.UUID) -> Asset:
    asset = await session.get(Asset, asset_id)
    if asset is None or asset.deleted_at is not None:
        raise NotFoundError(f"Asset '{asset_id}' not found")
    return asset


async def soft_delete_asset(session: AsyncSession, asset_id: uuid.UUID) -> None:
    asset = await get_asset(session, asset_id)
    asset.deleted_at = datetime.now(UTC)
    await session.commit()


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
) -> Asset:
    """Register a tool-produced file as a new (derived) Asset.

    Trusted output: skips the upload-size cap and allowed-extension gate that
    apply to user uploads. Sets source_asset_id for provenance / chaining.
    """
    stored_name = safe_filename(filename)
    extension = Path(filename).suffix.lstrip(".").lower()
    asset_id = uuid.uuid4()
    storage = get_storage()

    with src_path.open("rb") as src:
        storage_path, size_bytes = storage.save_upload(
            project_id, asset_id, stored_name, src
        )
    checksum = storage.calculate_checksum(storage_path)
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

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
    )
    session.add(asset)
    await session.commit()
    await session.refresh(asset)
    return asset
