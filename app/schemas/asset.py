"""Asset response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from app.models.enums import MediaType
from app.schemas.common import ORMModel


class AssetRead(ORMModel):
    id: uuid.UUID
    project_id: uuid.UUID
    filename: str
    original_filename: str
    media_type: MediaType
    mime_type: str
    extension: str
    size_bytes: int
    storage_path: str
    checksum_sha256: str
    source_asset_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
