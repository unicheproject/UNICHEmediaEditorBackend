"""Shared execution adapter for typed compose render requests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.composition.render import ComposeRenderRequest
from app.core.errors import ValidationError
from app.tools import ffmpeg
from app.tools.compose_policy import (
    DEFAULT_COMPOSE_RENDER_POLICY,
    ComposeRenderPolicy,
)


@dataclass(frozen=True)
class ComposeRenderSources:
    """Resolved media paths kept outside the deterministic semantic request."""

    video_paths: tuple[str, ...]
    audio_path: str | None = None
    subtitle_path: str | None = None


async def render_compose(
    request: ComposeRenderRequest,
    sources: ComposeRenderSources,
    dst: str,
    *,
    work_dir: Path,
    policy: ComposeRenderPolicy = DEFAULT_COMPOSE_RENDER_POLICY,
) -> None:
    """Execute canonical compose semantics through the existing FFmpeg tool."""
    if not request.video_asset_ids or len(request.video_asset_ids) != len(sources.video_paths):
        raise ValidationError("Compose render source count does not match its Asset references")
    if (request.audio_asset_id is None) != (sources.audio_path is None):
        raise ValidationError("Compose render audio source does not match its Asset reference")
    if (request.subtitle_asset_id is None) != (sources.subtitle_path is None):
        raise ValidationError("Compose render subtitle source does not match its Asset reference")
    await ffmpeg.compose(
        list(sources.video_paths),
        dst,
        width=request.width,
        height=request.height,
        work_dir=work_dir,
        audio_path=sources.audio_path,
        audio_mode=request.audio_mode.value,
        music_volume=request.music_volume,
        subtitle_path=sources.subtitle_path,
        policy=policy,
    )
