"""Pure schema-v1 CompositionRevision to typed render-spec compiler."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from pydantic import ValidationError as PydanticValidationError

from app.capabilities.compose_contract import (
    COMPOSE_DEFAULT_AUDIO_MODE,
    COMPOSE_DEFAULT_MUSIC_VOLUME,
    ComposeAudioMode,
)
from app.capabilities.context import JobContext
from app.composition.document import canonical_document
from app.core.errors import ValidationError
from app.models.composition import ProjectCompositionRevision
from app.schemas.composition import CompositionDocument

COMPOSITION_RENDER_COMPILER_VERSION = 1
COMPOSE_DEFAULT_WIDTH = 1920
COMPOSE_DEFAULT_HEIGHT = 1080


@dataclass(frozen=True)
class ComposeRenderRequest:
    """Semantic inputs to the existing compose renderer; contains no paths."""

    video_asset_ids: tuple[uuid.UUID, ...]
    width: int
    height: int
    audio_asset_id: uuid.UUID | None = None
    audio_mode: ComposeAudioMode = COMPOSE_DEFAULT_AUDIO_MODE
    music_volume: float = COMPOSE_DEFAULT_MUSIC_VOLUME
    subtitle_asset_id: uuid.UUID | None = None


@dataclass(frozen=True)
class ClipOccurrence:
    """Stable document identity for one ordered use of a source Asset."""

    clip_id: uuid.UUID
    asset_id: uuid.UUID


@dataclass(frozen=True)
class CompositionRenderSource:
    """Revision/audit identity kept separate from semantic render inputs."""

    composition_id: uuid.UUID
    revision_id: uuid.UUID
    revision_number: int
    schema_version: int
    document_hash: str
    compiler_version: int
    clip_occurrences: tuple[ClipOccurrence, ...]


@dataclass(frozen=True)
class CompiledCompositionRender:
    request: ComposeRenderRequest
    source: CompositionRenderSource


def _uuid(value: object, field_name: str) -> uuid.UUID | None:
    if value is None:
        return None
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValidationError(
            f"Invalid composition render reference for '{field_name}'",
            code="composition_invalid",
        ) from exc


def compose_request_from_job_context(ctx: JobContext) -> ComposeRenderRequest:
    """Translate an already-validated direct video.compose job to the shared type."""
    raw_mode = ctx.params.get("audio_mode", COMPOSE_DEFAULT_AUDIO_MODE.value)
    try:
        mode = ComposeAudioMode(raw_mode)
    except (ValueError, TypeError) as exc:
        raise ValidationError("Invalid video.compose audio mode") from exc
    raw_volume = ctx.params.get("music_volume", COMPOSE_DEFAULT_MUSIC_VOLUME)
    if isinstance(raw_volume, bool) or not isinstance(raw_volume, (int, float)):
        raise ValidationError("Invalid video.compose music volume")
    return ComposeRenderRequest(
        video_asset_ids=tuple(ctx.input_asset_ids),
        width=ctx.params.get("width", COMPOSE_DEFAULT_WIDTH),
        height=ctx.params.get("height", COMPOSE_DEFAULT_HEIGHT),
        audio_asset_id=_uuid(ctx.params.get("audio_asset_id"), "audio_asset_id"),
        audio_mode=mode,
        music_volume=float(raw_volume),
        subtitle_asset_id=_uuid(
            ctx.params.get("subtitle_asset_id"), "subtitle_asset_id"
        ),
    )


def compile_composition_revision(
    revision: ProjectCompositionRevision,
) -> CompiledCompositionRender:
    """Compile one immutable persisted revision without consulting its head."""
    try:
        document = CompositionDocument.model_validate(revision.document)
        occurrences = tuple(
            ClipOccurrence(clip_id=clip.id, asset_id=clip.asset_id)
            for track in document.tracks
            for clip in track.clips
        )
        video_ids = tuple(item.asset_id for item in occurrences)
        audio_id = document.audio_bed.asset_id if document.audio_bed else None
        if audio_id is not None and audio_id in video_ids:
            raise ValueError("An audio-bed Asset cannot also be a video clip")
        request = ComposeRenderRequest(
            video_asset_ids=video_ids,
            width=document.canvas.width,
            height=document.canvas.height,
            audio_asset_id=audio_id,
            audio_mode=(
                document.audio_bed.mode
                if document.audio_bed
                else COMPOSE_DEFAULT_AUDIO_MODE
            ),
            music_volume=(
                document.audio_bed.volume
                if document.audio_bed
                else COMPOSE_DEFAULT_MUSIC_VOLUME
            ),
        )
    except (PydanticValidationError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValidationError(
            "Saved composition revision is invalid", code="composition_invalid"
        ) from exc

    _, actual_hash = canonical_document(document)
    if actual_hash != revision.document_hash:
        raise ValidationError(
            "Saved composition revision integrity check failed",
            code="composition_invalid",
        )
    return CompiledCompositionRender(
        request=request,
        source=CompositionRenderSource(
            composition_id=revision.composition_id,
            revision_id=revision.id,
            revision_number=revision.revision_number,
            schema_version=document.schema_version,
            document_hash=revision.document_hash,
            compiler_version=COMPOSITION_RENDER_COMPILER_VERSION,
            clip_occurrences=occurrences,
        ),
    )
