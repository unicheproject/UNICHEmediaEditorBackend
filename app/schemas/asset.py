"""Asset response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import (
    AiProvenance,
    MediaType,
    ProvenanceKind,
    TechnicalMetadataStatus,
    WaveformReason,
    WaveformStatus,
)
from app.schemas.common import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, ORMModel


class AssetUpdate(BaseModel):
    original_filename: str | None = Field(default=None, min_length=1, max_length=512)


class AssetSearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str | None = Field(
        default=None, max_length=512,
        description=(
            "Literal case-insensitive substring of the display name (original_filename). "
            "Outer whitespace is ignored; empty means no name filter."
        ),
    )
    media_type: MediaType | None = Field(default=None, description="Omit to include all types.")
    sort_by: Literal["created_at", "name", "size", "type"] = Field(
        default="created_at",
        description=(
            "name: case-insensitive display name; size: bytes; type: alphabetical media type. "
            "Asset ID breaks ties in the selected direction."
        ),
    )
    sort_order: Literal["asc", "desc"] = "desc"
    limit: int = Field(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT)
    offset: int = Field(default=0, ge=0, le=2_147_483_647)
    include_intermediate: bool = Field(
        default=False, description="Include active intermediate/checkpoint Assets."
    )

    @field_validator("q")
    @classmethod
    def normalize_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if "\x00" in value:
            raise ValueError("Search text cannot contain null characters")
        return value.strip() or None


class AssetBulkDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_ids: list[uuid.UUID] = Field(
        min_length=1, max_length=100,
        description="Unique Asset IDs to soft-delete within the authorized project.",
        json_schema_extra={"uniqueItems": True},
    )

    @field_validator("asset_ids")
    @classmethod
    def unique_asset_ids(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        if len(value) != len(set(value)):
            raise ValueError("Asset IDs must be unique")
        return value


class AssetBulkDeleteItem(BaseModel):
    asset_id: uuid.UUID
    status: Literal["deleted", "not_found"] = Field(description=(
        "deleted: soft-deleted by this request; not_found: no active Asset with this "
        "ID in the authorized project, including already deleted or other-project IDs."
    ))


class AssetBulkDeleteResponse(BaseModel):
    results: list[AssetBulkDeleteItem] = Field(description=(
        "One outcome per requested ID, in request order. All matched Assets are "
        "soft-deleted in one transaction; binaries and audit records are retained."
    ))


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
    source_asset_id: uuid.UUID | None = Field(
        description="Immediate/best-effort input anchor; null does not imply Original."
    )
    provenance_kind: ProvenanceKind = Field(description=(
        "original: introduced as project source material, not produced by a recorded editor "
        "operation; derived: produced/materialized through a recorded editor operation; "
        "unknown: historical or insufficient evidence."
    ))
    ai_provenance: AiProvenance = Field(description=(
        "AI involvement recorded by this editor: none means no recorded AI processing, "
        "NOT certification of external human authorship; generated means direct AI content "
        "generation; modified means direct AI/model modification; inherited means ancestry "
        "from Assets with recorded AI involvement, not proof that AI content remains "
        "perceptually present; unknown means insufficient historical evidence."
    ))
    source_job_id: uuid.UUID | None = Field(description=(
        "Producing Job, or originating Job for a client-materialized/possibly edited "
        "derivative. Not byte-level attestation; null after Job deletion."
    ))
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
