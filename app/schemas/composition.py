"""Strict schema-v1 project composition document and API responses."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator

from app.capabilities.compose_contract import (
    COMPOSE_DEFAULT_AUDIO_MODE,
    COMPOSE_DEFAULT_MUSIC_VOLUME,
    IMAGE_CLIP_MAX_DURATION_SECONDS,
    IMAGE_CLIP_MIN_DURATION_SECONDS,
    ComposeAudioMode,
)

MAX_CANVAS_DIMENSION = 8192


class _StrictDocumentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CompositionCanvas(_StrictDocumentModel):
    width: int = Field(ge=2, le=MAX_CANVAS_DIMENSION)
    height: int = Field(ge=2, le=MAX_CANVAS_DIMENSION)

    @model_validator(mode="after")
    def require_even_dimensions(self) -> CompositionCanvas:
        if self.width % 2 or self.height % 2:
            raise ValueError("canvas width and height must be even for yuv420 output")
        return self


def _is_none(value: object) -> bool:
    return value is None


def _always(_value: object) -> bool:
    return True


class CompositionClip(_StrictDocumentModel):
    """A video clip. Its canonical form has no `kind`, as before image clips."""

    id: uuid.UUID = Field(strict=False)
    # Accepted for symmetry with image clips but never stored or returned, so a
    # video clip has one canonical form (and hash) with or without it.
    kind: Literal["video"] | None = Field(
        default=None, exclude_if=_always,
        description='Optional; "video" or omitted. Never returned.',
    )
    asset_id: uuid.UUID = Field(strict=False)
    # Optional source in/out points (not timeline positions), with video.trim's
    # semantics: seconds in the source Asset; an omitted or past-the-end
    # `source_end` keeps the rest of the clip.
    # Unset values are not serialized, so untimed clips (and the hashes of
    # revisions saved before these fields existed) are unchanged.
    source_start: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, exclude_if=_is_none,
        description="Optional in point: seconds into the source video (default 0).",
    )
    source_end: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, exclude_if=_is_none,
        description="Optional out point: seconds into the source video "
                    "(default: the end of the source).",
    )

    @model_validator(mode="after")
    def require_ordered_timing(self) -> CompositionClip:
        if self.source_end is not None and self.source_end <= (self.source_start or 0):
            raise ValueError("clip source_end must be after its source_start")
        return self

    @property
    def has_timing(self) -> bool:
        return self.source_start is not None or self.source_end is not None


class CompositionImageClip(_StrictDocumentModel):
    """A still image shown for `duration` seconds, scaled and padded to the canvas."""

    id: uuid.UUID = Field(strict=False)
    kind: Literal["image"]
    asset_id: uuid.UUID = Field(strict=False)
    duration: float = Field(
        ge=IMAGE_CLIP_MIN_DURATION_SECONDS,
        le=IMAGE_CLIP_MAX_DURATION_SECONDS,
        allow_inf_nan=False,
        description="Seconds the image is shown: at least one frame (1/30 s), at most 600.",
    )


def _clip_kind(value: Any) -> str | None:
    """Clips without `kind` are video clips, as in every document saved so far."""
    if isinstance(value, dict):
        kind = value.get("kind", "video")
        return kind if isinstance(kind, str) else None
    return "image" if isinstance(value, CompositionImageClip) else "video"


CompositionTrackClip = Annotated[
    Annotated[CompositionClip, Tag("video")] | Annotated[CompositionImageClip, Tag("image")],
    Discriminator(_clip_kind),
]


class CompositionVideoTrack(_StrictDocumentModel):
    id: uuid.UUID = Field(strict=False)
    kind: Literal["video"]
    clips: Annotated[list[CompositionTrackClip], Field(min_length=1)]


class CompositionAudioBed(_StrictDocumentModel):
    asset_id: uuid.UUID = Field(strict=False)
    mode: ComposeAudioMode = Field(default=COMPOSE_DEFAULT_AUDIO_MODE, strict=False)
    volume: float = Field(
        default=COMPOSE_DEFAULT_MUSIC_VOLUME,
        ge=0.0,
        le=1.0,
        description="Linear volume ratio in the inclusive range 0..1.",
    )


class CompositionDocument(_StrictDocumentModel):
    schema_version: Literal[1]
    canvas: CompositionCanvas
    tracks: Annotated[list[CompositionVideoTrack], Field(min_length=1, max_length=1)]
    audio_bed: CompositionAudioBed | None

    @model_validator(mode="after")
    def require_unique_clip_ids(self) -> CompositionDocument:
        clip_ids = [clip.id for track in self.tracks for clip in track.clips]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("clip IDs must be unique within the composition")
        return self


class CompositionRevisionRead(BaseModel):
    composition_id: uuid.UUID
    project_id: uuid.UUID
    revision_id: uuid.UUID
    revision_number: int
    schema_version: Literal[1]
    document_hash: str
    document: CompositionDocument
    created_at: datetime
