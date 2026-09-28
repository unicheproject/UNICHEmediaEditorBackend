"""On-demand thumbnail generation for image and video assets."""

from __future__ import annotations

import uuid
from pathlib import Path

from app.core.errors import UnsupportedMediaTypeError
from app.models.asset import Asset
from app.models.enums import MediaType
from app.services.storage import get_storage
from app.tools import ffmpeg, imagemagick

THUMBNAIL_FILENAME = "thumbnail.jpg"


async def get_or_create_thumbnail(asset: Asset) -> Path:
    """Return a cached, browser-safe JPEG preview for an image or video asset."""
    if asset.media_type not in {MediaType.image, MediaType.video}:
        raise UnsupportedMediaTypeError(
            "Thumbnails are only available for image and video assets"
        )

    source = get_storage().get_path(asset.storage_path)
    thumbnail_dir = source.parent.parent / "thumbnail"
    thumbnail_path = thumbnail_dir / THUMBNAIL_FILENAME
    if thumbnail_path.is_file():
        return thumbnail_path

    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = thumbnail_dir / f".{uuid.uuid4()}.jpg"
    try:
        if asset.media_type == MediaType.image:
            await imagemagick.thumbnail(str(source), str(temporary_path))
        else:
            await ffmpeg.video_thumbnail(str(source), str(temporary_path), timestamp=0)
        temporary_path.replace(thumbnail_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return thumbnail_path
