"""Asset ORM model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import false as sa_false

from app.models.base import Base, JSONType, SoftDeleteMixin, TimestampMixin
from app.models.enums import MediaType, TechnicalMetadataStatus, WaveformStatus

if TYPE_CHECKING:
    from app.models.project import Project


class Asset(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "assets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )

    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    media_type: Mapped[MediaType] = mapped_column(
        Enum(MediaType, name="media_type"), nullable=False
    )
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    extension: Mapped[str] = mapped_column(String(32), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    # Provenance: null = uploaded original; set = derived from a capability job.
    source_asset_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("assets.id"), nullable=True, index=True
    )
    # True when produced as an intermediate step of an agent plan (consumed by a
    # later step). Finals/originals are False. Lets the UI hide scaffolding.
    is_intermediate: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sa_false(), index=True
    )
    # Temp Processed Assets retain their audit row after the binary expires.
    # Historical intermediates keep NULL here and are not automatically swept.
    temp_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    physical_expired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Canonical, normalized technical media metadata. Summary columns support
    # common queries; technical_streams stores a bounded application-owned
    # projection of relevant per-stream facts, never raw ffprobe output.
    technical_metadata_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=TechnicalMetadataStatus.pending.value,
        server_default=TechnicalMetadataStatus.pending.value,
        index=True,
    )
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    container_format: Mapped[str | None] = mapped_column(String(128), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    has_video_stream: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    has_audio_stream: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    video_codec: Mapped[str | None] = mapped_column(String(128), nullable=True)
    audio_codec: Mapped[str | None] = mapped_column(String(128), nullable=True)
    frame_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    audio_channels: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bit_rate: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    technical_streams: Mapped[dict | None] = mapped_column(JSONType, nullable=True)

    waveform_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=WaveformStatus.pending.value,
        server_default=WaveformStatus.pending.value,
    )
    waveform_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)

    project: Mapped[Project] = relationship(
        back_populates="assets", lazy="noload"
    )
