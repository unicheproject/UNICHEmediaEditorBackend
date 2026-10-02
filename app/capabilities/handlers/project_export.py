"""Asynchronous renderer for an immutable saved project composition."""

from __future__ import annotations

from app.capabilities.context import HandlerResult, JobContext, OutputFile
from app.capabilities.handlers.base import LocalToolHandler
from app.core.errors import ValidationError
from app.models.enums import MediaType
from app.tools import ffmpeg
from app.tools.compose_policy import (
    DEFAULT_COMPOSE_RENDER_POLICY,
    LOSSLESS_COMPOSE_RENDER_POLICY,
)
from app.tools.compose_renderer import render_compose
from app.tools.export_policy import CompositionPolicyId, EffectiveExportSettings


class ProjectExportHandler(LocalToolHandler):
    capability_id = "project.export"

    async def process(self, ctx: JobContext) -> HandlerResult:
        execution = ctx.project_export
        if execution is None or ctx.work_dir is None:
            raise ValidationError("Project export execution context is unavailable")

        settings: EffectiveExportSettings = execution.settings
        revision_number = execution.compiled.source.revision_number
        preset = execution.snapshot.preset.value
        filename = f"project-export-r{revision_number}-{preset}.{settings.container.value}"
        output = OutputFile(
            path=ctx.out_path(filename),
            filename=filename,
            media_type=MediaType.video,
        )
        request = execution.compiled.request

        if settings.composition_policy == CompositionPolicyId.LOSSLESS:
            # FFV1/FLAC is selected from the first generated segment onward;
            # there is no default H.264/AAC composition intermediate.
            await render_compose(
                request,
                execution.sources,
                str(output.path),
                work_dir=ctx.work_dir,
                policy=LOSSLESS_COMPOSE_RENDER_POLICY,
            )
            video_passes = len(request.video_asset_ids)
        else:
            composition_path = ctx.out_path("_composition-default.mp4")
            await render_compose(
                request,
                execution.sources,
                str(composition_path),
                work_dir=ctx.work_dir,
                policy=DEFAULT_COMPOSE_RENDER_POLICY,
            )
            await ffmpeg.export_delivery_transform(
                str(composition_path), str(output.path), settings
            )
            video_passes = len(request.video_asset_ids) + 1

        return HandlerResult(
            data={
                "operation": self.capability_id,
                "composition_id": str(execution.compiled.source.composition_id),
                "composition_revision_id": str(execution.compiled.source.revision_id),
                "preset": preset,
                "video_encode_passes": video_passes,
            },
            outputs=[output],
        )
