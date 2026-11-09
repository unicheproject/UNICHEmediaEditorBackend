"""Built-in project templates (MED/028): versioned JSON over the composition schema.

A template is a composition document whose clips name a *slot* instead of an
Asset. Applying it binds each slot to Assets of the target project and saves an
ordinary composition revision; no new timeline representation exists.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator

from app.capabilities.compose_contract import (
    AUDIO_TRACK_MAX_CLIPS,
    COMPOSE_DEFAULT_AUDIO_MODE,
    COMPOSE_DEFAULT_MUSIC_VOLUME,
    IMAGE_CLIP_MAX_DURATION_SECONDS,
    IMAGE_CLIP_MIN_DURATION_SECONDS,
    ComposeAudioMode,
)
from app.composition.export_presets import ExportPresetId
from app.schemas.composition import CompositionCanvas

TEMPLATE_ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
SLOT_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,39}$"
TEMPLATE_SLOT_MAX_ASSETS = 12

SlotName = Annotated[str, Field(pattern=SLOT_NAME_PATTERN)]


class _StrictTemplateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CapabilitySuggestion(_StrictTemplateModel):
    """Informational only: how a client could create an Asset for the slot."""

    capability_id: Literal["media.titlecard"]
    params: dict[str, str | int | float] = Field(default_factory=dict)


class TemplateSlot(_StrictTemplateModel):
    media_type: Literal["video", "image", "audio"]
    required: bool
    multiple: bool = False
    max: int | None = Field(default=None, ge=1, le=TEMPLATE_SLOT_MAX_ASSETS)
    label: str = Field(min_length=1, max_length=80)
    help: str | None = Field(default=None, max_length=300)
    suggestion: CapabilitySuggestion | None = None

    @model_validator(mode="after")
    def require_consistent_cardinality(self) -> TemplateSlot:
        if self.max is not None and not self.multiple:
            raise ValueError("only multiple slots take max")
        if self.suggestion is not None and self.media_type != "video":
            raise ValueError("titlecard suggestions apply to video slots")
        return self

    @property
    def max_assets(self) -> int:
        return (self.max or TEMPLATE_SLOT_MAX_ASSETS) if self.multiple else 1


class TemplateVideoClip(_StrictTemplateModel):
    slot: SlotName
    source_start: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    source_end: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class TemplateImageClip(_StrictTemplateModel):
    slot: SlotName
    kind: Literal["image"]
    duration: float = Field(
        ge=IMAGE_CLIP_MIN_DURATION_SECONDS, le=IMAGE_CLIP_MAX_DURATION_SECONDS,
        allow_inf_nan=False,
    )


def _visual_kind(value: Any) -> str | None:
    if isinstance(value, dict):
        kind = value.get("kind", "video")
        return kind if isinstance(kind, str) else None
    return "image" if isinstance(value, TemplateImageClip) else "video"


TemplateVisualClip = Annotated[
    Annotated[TemplateVideoClip, Tag("video")] | Annotated[TemplateImageClip, Tag("image")],
    Discriminator(_visual_kind),
]


class TemplateAudioClip(_StrictTemplateModel):
    slot: SlotName
    timeline_start: float = Field(ge=0, allow_inf_nan=False)
    source_start: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    source_end: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    volume: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)


class TemplateVideoTrack(_StrictTemplateModel):
    kind: Literal["video"]
    clips: Annotated[list[TemplateVisualClip], Field(min_length=1)]


class TemplateAudioTrack(_StrictTemplateModel):
    kind: Literal["audio"]
    clips: Annotated[
        list[TemplateAudioClip], Field(min_length=1, max_length=AUDIO_TRACK_MAX_CLIPS)
    ]


TemplateTrack = Annotated[TemplateVideoTrack | TemplateAudioTrack, Field(discriminator="kind")]


class TemplateAudioBed(_StrictTemplateModel):
    slot: SlotName
    mode: ComposeAudioMode = Field(default=COMPOSE_DEFAULT_AUDIO_MODE, strict=False)
    volume: float = Field(default=COMPOSE_DEFAULT_MUSIC_VOLUME, ge=0.0, le=1.0)


class CompositionTemplate(_StrictTemplateModel):
    template_schema: Literal[1]
    template_id: str = Field(pattern=TEMPLATE_ID_PATTERN, max_length=64)
    version: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=400)
    canvas: CompositionCanvas
    slots: Annotated[dict[SlotName, TemplateSlot], Field(min_length=1)]
    tracks: Annotated[list[TemplateTrack], Field(min_length=1, max_length=2)]
    audio_bed: TemplateAudioBed | None = None
    suggested_export_preset: ExportPresetId | None = Field(default=None, strict=False)

    @model_validator(mode="after")
    def require_consistent_slots(self) -> CompositionTemplate:
        kinds = [track.kind for track in self.tracks]
        if kinds not in (["video"], ["video", "audio"]):
            raise ValueError("tracks must be one video track, optionally followed by "
                             "one audio track")
        uses: list[tuple[str, str]] = []
        for track in self.tracks:
            for clip in track.clips:
                media = (
                    "image" if isinstance(clip, TemplateImageClip)
                    else "audio" if isinstance(clip, TemplateAudioClip) else "video"
                )
                uses.append((clip.slot, media))
        if self.audio_bed is not None:
            uses.append((self.audio_bed.slot, "audio"))
        names = [name for name, _ in uses]
        if len(names) != len(set(names)):
            raise ValueError("each slot is used exactly once")
        if set(names) != set(self.slots):
            raise ValueError("every slot must be used, and every used slot defined")
        for name, media in uses:
            if self.slots[name].media_type != media:
                raise ValueError(f"slot {name!r} is used as {media}")
        if self.audio_bed is not None and self.slots[self.audio_bed.slot].multiple:
            raise ValueError("the audio bed takes a single Asset")
        visual = self.tracks[0].clips
        if not any(self.slots[clip.slot].required for clip in visual):
            raise ValueError("a required slot must fill the video track")
        return self


class CompositionTemplateSummary(BaseModel):
    template_id: str
    version: int
    name: str
    description: str
    slots: dict[str, TemplateSlot]
    suggested_export_preset: ExportPresetId | None


class ApplyTemplateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str = Field(pattern=TEMPLATE_ID_PATTERN, max_length=64)
    version: int = Field(ge=1, strict=True)
    bindings: dict[str, uuid.UUID | list[uuid.UUID] | None] = Field(
        default_factory=dict,
        description="Slot name -> Asset ID, or a list of Asset IDs for multiple slots; "
                    "optional slots may be omitted or null.",
    )
