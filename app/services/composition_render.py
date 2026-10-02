"""Runtime resolution and revalidation for an immutable composition render."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.composition.render import CompiledCompositionRender, compile_composition_revision
from app.core.errors import AppError, NotFoundError, ValidationError
from app.models.asset import Asset
from app.models.composition import ProjectComposition, ProjectCompositionRevision
from app.models.enums import MediaType
from app.services.assets import ensure_asset_binary_available
from app.services.projects import get_project
from app.tools.compose_renderer import ComposeRenderSources

_UNAVAILABLE = "A saved composition references media that is unavailable or invalid"


@dataclass(frozen=True)
class ResolvedCompositionRender:
    compiled: CompiledCompositionRender
    project_id: uuid.UUID
    sources: ComposeRenderSources


async def resolve_composition_render(
    session: AsyncSession,
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
) -> ResolvedCompositionRender:
    """Load one project-owned revision and revalidate all media at render time."""
    await get_project(session, project_id)
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
        raise NotFoundError("Composition revision not found", code="composition_revision_not_found")
    _composition, revision = row
    compiled = compile_composition_revision(revision)

    expected: dict[uuid.UUID, MediaType] = dict.fromkeys(
        compiled.request.video_asset_ids, MediaType.video
    )
    audio_id = compiled.request.audio_asset_id
    if audio_id is not None:
        if audio_id in expected:
            raise ValidationError(_UNAVAILABLE, code="composition_asset_unavailable")
        expected[audio_id] = MediaType.audio

    result_assets = await session.scalars(
        select(Asset).where(Asset.id.in_(expected)).with_for_update()
    )
    assets = {asset.id: asset for asset in result_assets}
    paths: dict[uuid.UUID, str] = {}
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
            raise ValidationError(_UNAVAILABLE, code="composition_asset_unavailable")
        try:
            paths[asset_id] = str(ensure_asset_binary_available(asset))
        except (AppError, OSError) as exc:
            raise ValidationError(
                _UNAVAILABLE, code="composition_asset_unavailable"
            ) from exc

    audio_path = str(paths[audio_id]) if audio_id is not None else None
    return ResolvedCompositionRender(
        compiled=compiled,
        project_id=project_id,
        sources=ComposeRenderSources(
            video_paths=tuple(paths[asset_id] for asset_id in compiled.request.video_asset_ids),
            audio_path=audio_path,
        ),
    )
