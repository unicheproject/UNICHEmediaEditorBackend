"""Strict schema-v1 project composition document and API responses."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.capabilities.compose_contract import (
    COMPOSE_DEFAULT_AUDIO_MODE,
    COMPOSE_DEFAULT_MUSIC_VOLUME,
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


class CompositionClip(_StrictDocumentModel):
    id: uuid.UUID = Field(strict=False)
    asset_id: uuid.UUID = Field(strict=False)


class CompositionVideoTrack(_StrictDocumentModel):
    id: uuid.UUID = Field(strict=False)
    kind: Literal["video"]
    clips: Annotated[list[CompositionClip], Field(min_length=1)]


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
