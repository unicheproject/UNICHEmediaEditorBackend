"""Persistence, validation, and concurrency for project compositions."""

from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.composition.document import canonical_document
from app.core.errors import (
    BadRequestError,
    NotFoundError,
    PreconditionFailedError,
    PreconditionRequiredError,
    ValidationError,
)
from app.models.asset import Asset
from app.models.composition import ProjectComposition, ProjectCompositionRevision
from app.models.enums import MediaType, TechnicalMetadataStatus
from app.schemas.composition import CompositionDocument, CompositionImageClip
from app.services.assets import ensure_asset_binary_available

_INVALID_ASSET_REFERENCE = (
    "Composition references must be available, durable Assets of the expected "
    "media type in this project"
)


def etag_for_revision(revision_id: uuid.UUID) -> str:
    return f'"{revision_id}"'


def _parse_revision_etag(value: str | None) -> uuid.UUID:
    if value is None:
        raise PreconditionRequiredError("If-Match is required to update a composition")
    if len(value) < 3 or value[0] != '"' or value[-1] != '"':
        raise BadRequestError("If-Match must contain one strong composition ETag")
    try:
        parsed = uuid.UUID(value[1:-1])
    except ValueError as exc:
        raise BadRequestError("If-Match is not a valid composition ETag") from exc
    if str(parsed) != value[1:-1]:
        raise BadRequestError("If-Match must use the canonical composition ETag")
    return parsed


async def _validate_asset_references(
    session: AsyncSession, project_id: uuid.UUID, document: CompositionDocument
) -> None:
    expected: dict[uuid.UUID, MediaType] = {}
    for clip in document.video_track.clips:
        media_type = (
            MediaType.image if isinstance(clip, CompositionImageClip) else MediaType.video
        )
        if expected.setdefault(clip.asset_id, media_type) != media_type:
            raise ValidationError(_INVALID_ASSET_REFERENCE)
    if document.audio_bed is not None:
        aid = document.audio_bed.asset_id
        if aid in expected:
            raise ValidationError("An Asset cannot be used as both a video clip and audio bed")
        expected[aid] = MediaType.audio
    audio_track = document.audio_track
    for audio_clip in audio_track.clips if audio_track is not None else ():
        if expected.setdefault(audio_clip.asset_id, MediaType.audio) != MediaType.audio:
            raise ValidationError(_INVALID_ASSET_REFERENCE)

    result = await session.scalars(
        select(Asset).where(Asset.id.in_(expected)).with_for_update()
    )
    assets = {asset.id: asset for asset in result}
    for asset_id, media_type in expected.items():
        asset = assets.get(asset_id)
        if (
            asset is None
            or asset.project_id != project_id
            or asset.deleted_at is not None
            or asset.media_type != media_type
            or asset.is_intermediate
            or asset.temp_expires_at is not None
            or asset.physical_expired_at is not None
        ):
            raise ValidationError(_INVALID_ASSET_REFERENCE)
        try:
            ensure_asset_binary_available(asset)
        except (NotFoundError, ValidationError) as exc:
            raise ValidationError(_INVALID_ASSET_REFERENCE) from exc

    # Clip in/out points: like video.trim admission, check against already-probed
    # metadata when available; otherwise the render checks the source itself.
    from app.tools import ffmpeg

    # Image clips: the same applies to the decoded-size limit.
    # The video track's length is estimated only when every clip's length is
    # known. Container durations can only overstate a video stream, so this
    # estimate never rejects an audio clip that the render would accept.
    visual_seconds: float | None = 0.0
    for clip in document.video_track.clips:
        asset = assets[clip.asset_id]
        known = asset.technical_metadata_status == TechnicalMetadataStatus.available.value
        if isinstance(clip, CompositionImageClip):
            if known:
                ffmpeg.check_still_image_pixels(asset.width, asset.height)
            if visual_seconds is not None:
                visual_seconds += clip.duration
            continue
        if known and clip.has_timing:
            ffmpeg.resolve_trim_window(
                clip.source_start or 0.0, clip.source_end,
                duration=asset.duration_seconds, frame_rate=asset.frame_rate,
            )
        if not known or not asset.duration_seconds or visual_seconds is None:
            visual_seconds = None
            continue
        end = asset.duration_seconds
        if clip.source_end is not None:
            end = min(end, clip.source_end)
        visual_seconds += end - (clip.source_start or 0.0)

    for audio_clip in audio_track.clips if audio_track is not None else ():
        asset = assets[audio_clip.asset_id]
        if asset.technical_metadata_status == TechnicalMetadataStatus.available.value:
            ffmpeg.resolve_trim_window(
                audio_clip.source_start or 0.0, audio_clip.source_end,
                duration=asset.duration_seconds, frame_rate=None, media="audio",
            )
        ffmpeg.check_audio_clip_start(audio_clip.timeline_start, visual_seconds)


def _revision_read(
    composition: ProjectComposition, revision: ProjectCompositionRevision
) -> dict:
    return {
        "composition_id": composition.id,
        "project_id": composition.project_id,
        "revision_id": revision.id,
        "revision_number": revision.revision_number,
        "schema_version": revision.document["schema_version"],
        "document_hash": revision.document_hash,
        "document": revision.document,
        "created_at": revision.created_at,
    }


async def get_current_composition(
    session: AsyncSession, project_id: uuid.UUID
) -> tuple[dict, str]:
    composition = await session.scalar(
        select(ProjectComposition).where(ProjectComposition.project_id == project_id)
    )
    if composition is None:
        raise NotFoundError("Project has no saved composition", code="composition_not_found")
    revision = await session.scalar(
        select(ProjectCompositionRevision).where(
            ProjectCompositionRevision.composition_id == composition.id,
            ProjectCompositionRevision.revision_number == composition.current_revision_number,
        )
    )
    if revision is None:
        raise NotFoundError("Current composition revision not found")
    response = _revision_read(composition, revision)
    return response, etag_for_revision(revision.id)


async def get_composition_revision(
    session: AsyncSession, project_id: uuid.UUID, revision_id: uuid.UUID
) -> dict:
    result = await session.execute(
        select(ProjectComposition, ProjectCompositionRevision)
        .join(
            ProjectCompositionRevision,
            ProjectCompositionRevision.composition_id == ProjectComposition.id,
        )
        .where(
            ProjectComposition.project_id == project_id,
            ProjectCompositionRevision.id == revision_id,
        )
    )
    row = result.one_or_none()
    if row is None:
        raise NotFoundError("Composition revision not found")
    composition, revision = row
    return _revision_read(composition, revision)


async def save_composition(
    session: AsyncSession,
    project_id: uuid.UUID,
    document: CompositionDocument,
    *,
    author_subject: str | None,
    if_match: str | None,
    if_none_match: str | None,
) -> tuple[dict, str, bool]:
    """Conditionally save a document; return response, ETag, and created flag."""
    if if_match is not None and if_none_match is not None:
        raise BadRequestError("Supply only one of If-Match or If-None-Match")
    normalized, document_hash = canonical_document(document)

    composition = await session.scalar(
        select(ProjectComposition)
        .where(ProjectComposition.project_id == project_id)
        .with_for_update()
    )
    if composition is None:
        if if_none_match is None:
            raise PreconditionRequiredError(
                "If-None-Match: * is required to create a composition"
            )
        if if_none_match.strip() != "*":
            raise BadRequestError("First save requires If-None-Match: *")
        await _validate_asset_references(session, project_id, document)
        composition = ProjectComposition(
            id=uuid.uuid4(), project_id=project_id, current_revision_number=1
        )
        revision = ProjectCompositionRevision(
            id=uuid.uuid4(),
            composition_id=composition.id,
            revision_number=1,
            document=normalized,
            document_hash=document_hash,
            author_subject=author_subject,
        )
        session.add_all([composition, revision])
        try:
            await session.commit()
            await session.refresh(composition)
            await session.refresh(revision)
        except IntegrityError as exc:
            await session.rollback()
            raise PreconditionFailedError(
                "A composition was created concurrently; reload before saving"
            ) from exc
        return _revision_read(composition, revision), etag_for_revision(revision.id), True

    if if_none_match is not None:
        raise PreconditionFailedError("A project composition already exists")
    supplied_revision_id = _parse_revision_etag(if_match)
    current_revision = await session.scalar(
        select(ProjectCompositionRevision).where(
            ProjectCompositionRevision.composition_id == composition.id,
            ProjectCompositionRevision.revision_number == composition.current_revision_number,
        )
    )
    if current_revision is None or supplied_revision_id != current_revision.id:
        raise PreconditionFailedError("Composition changed; reload before saving")

    await _validate_asset_references(session, project_id, document)
    if current_revision.document_hash == document_hash:
        response = _revision_read(composition, current_revision)
        etag = etag_for_revision(current_revision.id)
        await session.rollback()
        return response, etag, False

    next_number = composition.current_revision_number + 1
    cas = await session.execute(
        update(ProjectComposition)
        .where(
            ProjectComposition.id == composition.id,
            ProjectComposition.current_revision_number == composition.current_revision_number,
        )
        .values(current_revision_number=next_number)
        .returning(ProjectComposition.id)
    )
    if cas.scalar_one_or_none() is None:
        await session.rollback()
        raise PreconditionFailedError("Composition changed; reload before saving")

    revision = ProjectCompositionRevision(
        id=uuid.uuid4(),
        composition_id=composition.id,
        revision_number=next_number,
        document=normalized,
        document_hash=document_hash,
        author_subject=author_subject,
    )
    session.add(revision)
    try:
        await session.commit()
        await session.refresh(revision)
    except IntegrityError as exc:
        await session.rollback()
        raise PreconditionFailedError("Composition changed; reload before saving") from exc
    composition.current_revision_number = next_number
    return _revision_read(composition, revision), etag_for_revision(revision.id), True
