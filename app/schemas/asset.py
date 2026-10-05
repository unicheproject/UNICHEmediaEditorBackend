"""Asset response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import MediaType, TechnicalMetadataStatus, WaveformReason, WaveformStatus
from app.schemas.common import ORMModel


class AssetUpdate(BaseModel):
    original_filename: str | None = Field(default=None, min_length=1, max_length=512)


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
    is_intermediate: bool
    temp_expires_at: datetime | None
    physical_expired_at: datetime | None
    technical_metadata_status: TechnicalMetadataStatus
    duration_seconds: float | None
    container_format: str | None
    width: int | None
    height: int | None
    has_video_stream: bool | None
    has_audio_stream: bool | None
    video_codec: str | None
    audio_codec: str | None
    frame_rate: float | None
    audio_sample_rate: int | None
    audio_channels: int | None
    bit_rate: int | None
    technical_streams: dict | None
    waveform_status: WaveformStatus
    waveform_reason: WaveformReason | None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
