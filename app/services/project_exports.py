"""Submission snapshots and worker-time source resolution for project exports."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.composition.export_presets import (
    EXPORT_PRESET_VERSION,
    ExportPresetId,
    resolve_effective_settings,
)
from app.composition.render import CompiledCompositionRender
from app.core.errors import ValidationError
from app.services.composition_render import (
    ResolvedCompositionRender,
    resolve_composition_render,
)
from app.tools.compose_renderer import ComposeRenderSources
from app.tools.export_policy import EffectiveExportSettings

SNAPSHOT_KEY = "execution_snapshot"
EXPORT_SNAPSHOT_VERSION = 1


class AssetContentIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: uuid.UUID
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExportExecutionSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_version: int
    composition_revision_id: uuid.UUID
    composition_document_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    composition_compiler_version: int
    preset: ExportPresetId
    preset_version: int
    selection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_video_sources: tuple[AssetContentIdentity, ...]
    audio_source: AssetContentIdentity | None
    effective_settings: EffectiveExportSettings


@dataclass(frozen=True)
class ProjectExportExecution:
    compiled: CompiledCompositionRender
    sources: ComposeRenderSources
    settings: EffectiveExportSettings
    snapshot: ExportExecutionSnapshot


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selection_hash(request: dict[str, Any]) -> str:
    return _canonical_hash(request)


async def create_export_execution_snapshot(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Validate an exact revision and persist its resolved settings in Job input."""
    try:
        revision_id = uuid.UUID(str(request["composition_revision_id"]))
        resolved = await resolve_composition_render(session, project_id, revision_id)
        settings = resolve_effective_settings(
            request,
            canvas_width=resolved.compiled.request.width,
            canvas_height=resolved.compiled.request.height,
        )
        snapshot = ExportExecutionSnapshot(
            snapshot_version=EXPORT_SNAPSHOT_VERSION,
            composition_revision_id=revision_id,
            composition_document_hash=resolved.compiled.source.document_hash,
            composition_compiler_version=resolved.compiled.source.compiler_version,
            preset=ExportPresetId(request["preset"]),
            preset_version=EXPORT_PRESET_VERSION,
            selection_hash=_selection_hash(request),
            ordered_video_sources=tuple(
                AssetContentIdentity(
                    asset_id=asset_id,
                    checksum_sha256=checksum,
                )
                for asset_id, checksum in zip(
                    resolved.compiled.request.video_asset_ids,
                    resolved.ordered_video_checksums,
                    strict=True,
                )
            ),
            audio_source=(
                AssetContentIdentity(
                    asset_id=resolved.compiled.request.audio_asset_id,
                    checksum_sha256=resolved.audio_checksum,
                )
                if (
                    resolved.compiled.request.audio_asset_id is not None
                    and resolved.audio_checksum is not None
                )
                else None
            ),
            effective_settings=settings,
        )
    except (KeyError, ValueError, TypeError, PydanticValidationError) as exc:
        raise ValidationError(f"Invalid project.export request: {exc}") from exc
    return snapshot.model_dump(mode="json")


async def resolve_project_export_execution(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    job_input: dict[str, Any],
) -> ProjectExportExecution:
    """Revalidate snapshot, immutable revision, source binaries and checksums."""
    raw_snapshot = job_input.get(SNAPSHOT_KEY)
    if not isinstance(raw_snapshot, dict):
        raise ValidationError("Project export execution snapshot is missing")
    try:
        snapshot = ExportExecutionSnapshot.model_validate(raw_snapshot)
        revision_id = uuid.UUID(str(job_input["composition_revision_id"]))
        if (
            snapshot.snapshot_version != EXPORT_SNAPSHOT_VERSION
            or snapshot.composition_revision_id != revision_id
            or snapshot.preset != ExportPresetId(job_input["preset"])
            or snapshot.preset_version != EXPORT_PRESET_VERSION
            or snapshot.selection_hash != _selection_hash(
                {key: value for key, value in job_input.items() if key != SNAPSHOT_KEY}
            )
        ):
            raise ValueError("Project export Job input no longer matches its snapshot")
    except (KeyError, ValueError, TypeError, PydanticValidationError) as exc:
        raise ValidationError("Project export execution snapshot is invalid") from exc

    resolved: ResolvedCompositionRender = await resolve_composition_render(
        session, project_id, revision_id
    )
    compiled = resolved.compiled
    if (
        compiled.source.document_hash != snapshot.composition_document_hash
        or compiled.source.compiler_version != snapshot.composition_compiler_version
    ):
        raise ValidationError("Composition revision no longer matches the export snapshot")

    current_video_sources = tuple(
        AssetContentIdentity(asset_id=asset_id, checksum_sha256=checksum)
        for asset_id, checksum in zip(
            compiled.request.video_asset_ids,
            resolved.ordered_video_checksums,
            strict=True,
        )
    )
    current_audio_source = (
        AssetContentIdentity(
            asset_id=compiled.request.audio_asset_id,
            checksum_sha256=resolved.audio_checksum,
        )
        if compiled.request.audio_asset_id is not None and resolved.audio_checksum
        else None
    )
    if (
        current_video_sources != snapshot.ordered_video_sources
        or current_audio_source != snapshot.audio_source
    ):
        raise ValidationError("Composition source content does not match the export snapshot")

    try:
        # Execute exactly the settings snapshotted at submission. The worker
        # validates their closed typed shape but does not resolve current preset
        # definitions again.
        settings = EffectiveExportSettings.model_validate(
            snapshot.effective_settings.model_dump(mode="python")
        )
        if snapshot.preset == ExportPresetId.ARCHIVAL:
            if settings.composition_policy.value != "lossless":
                raise ValueError("Archival snapshot is not lossless")
        elif settings.composition_policy.value == "lossless":
            raise ValueError("Only the locked Archival preset may use lossless mode")
        if snapshot.preset == ExportPresetId.SOCIAL:
            expected_target = (
                (1080, 1080)
                if job_input.get("social_format") == "square"
                else (1080, 1920)
            )
            if (settings.target_width, settings.target_height) != expected_target:
                raise ValueError("Social snapshot canvas does not match its selected format")
    except (ValueError, TypeError, PydanticValidationError) as exc:
        raise ValidationError("Snapshotted project export settings are invalid") from exc

    return ProjectExportExecution(
        compiled=compiled,
        sources=resolved.sources,
        settings=settings,
        snapshot=snapshot,
    )
