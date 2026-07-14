"""Deterministic video capability handlers (FFmpeg)."""

from __future__ import annotations

from app.capabilities.context import HandlerResult, JobContext, OutputFile
from app.capabilities.handlers._util import (
    input_ext,
    input_stem,
    require,
    require_input,
)
from app.capabilities.handlers.base import LocalToolHandler
from app.core.errors import ValidationError
from app.models.enums import MediaType
from app.tools import ffmpeg, pyscenedetect


def _video_out(ctx: JobContext, suffix: str, ext: str | None = None) -> OutputFile:
    ext = ext or input_ext(ctx)
    filename = f"{input_stem(ctx)}_{suffix}.{ext}"
    return OutputFile(path=ctx.out_path(filename), filename=filename, media_type=MediaType.video)


class VideoTrimHandler(LocalToolHandler):
    capability_id = "video.trim"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _video_out(ctx, "trim")
        await ffmpeg.video_trim(
            src, str(out.path), require(ctx.params, "start"), require(ctx.params, "end")
        )
        return HandlerResult(data={"operation": "video.trim"}, outputs=[out])


class VideoSplitHandler(LocalToolHandler):
    capability_id = "video.split"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        markers = require(ctx.params, "markers")
        if not isinstance(markers, list) or not markers:
            raise ValidationError("'markers' must be a non-empty list of seconds")
        ext = input_ext(ctx)
        pattern = str(ctx.out_path(f"{input_stem(ctx)}_seg%03d.{ext}"))
        await ffmpeg.video_split(src, pattern, markers)
        files = ffmpeg.segment_outputs(pattern)
        if not files:
            raise ValidationError("Split produced no segments")
        outputs = [
            OutputFile(path=p, filename=p.name, media_type=MediaType.video)
            for p in files
        ]
        return HandlerResult(
            data={"operation": "video.split", "segments": len(outputs)},
            outputs=outputs,
        )


class VideoConcatHandler(LocalToolHandler):
    capability_id = "video.concat"

    async def process(self, ctx: JobContext) -> HandlerResult:
        if len(ctx.input_paths) < 2:
            raise ValidationError("video.concat needs at least 2 input assets (input.asset_ids)")
        assert ctx.work_dir is not None
        ext = input_ext(ctx)
        out = OutputFile(
            path=ctx.out_path(f"{input_stem(ctx)}_concat.{ext}"),
            filename=f"{input_stem(ctx)}_concat.{ext}",
            media_type=MediaType.video,
        )
        await ffmpeg.video_concat(ctx.input_paths, str(out.path), ctx.work_dir)
        return HandlerResult(
            data={"operation": "video.concat", "inputs": len(ctx.input_paths)},
            outputs=[out],
        )


class VideoTranscodeHandler(LocalToolHandler):
    capability_id = "video.transcode"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        fmt = str(require(ctx.params, "format")).lstrip(".").lower()
        out = _video_out(ctx, "transcode", ext=fmt)
        await ffmpeg.video_transcode(
            src, str(out.path), ctx.params.get("video_codec"), ctx.params.get("audio_codec")
        )
        return HandlerResult(data={"operation": "video.transcode", "format": fmt}, outputs=[out])


class VideoMuteHandler(LocalToolHandler):
    capability_id = "video.mute"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _video_out(ctx, "muted")
        await ffmpeg.video_mute(src, str(out.path))
        return HandlerResult(data={"operation": "video.mute"}, outputs=[out])


class VideoCropHandler(LocalToolHandler):
    capability_id = "video.crop"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _video_out(ctx, "crop")
        await ffmpeg.video_crop(
            src, str(out.path),
            require(ctx.params, "x"), require(ctx.params, "y"),
            require(ctx.params, "width"), require(ctx.params, "height"),
        )
        return HandlerResult(data={"operation": "video.crop"}, outputs=[out])


class VideoResizeHandler(LocalToolHandler):
    capability_id = "video.resize"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _video_out(ctx, "resize")
        await ffmpeg.video_resize(
            src, str(out.path), require(ctx.params, "width"), require(ctx.params, "height")
        )
        return HandlerResult(data={"operation": "video.resize"}, outputs=[out])


class VideoThumbnailHandler(LocalToolHandler):
    capability_id = "video.thumbnail"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        filename = f"{input_stem(ctx)}_thumb.jpg"
        out = OutputFile(
            path=ctx.out_path(filename), filename=filename, media_type=MediaType.image
        )
        await ffmpeg.video_thumbnail(src, str(out.path), require(ctx.params, "timestamp"))
        return HandlerResult(data={"operation": "video.thumbnail"}, outputs=[out])


class VideoShotDetectHandler(LocalToolHandler):
    """Detects shot/scene boundaries (PySceneDetect ContentDetector). JSON-only: no output file."""

    capability_id = "video.shot.detect"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        shots = await pyscenedetect.detect_shots(src, ctx.params.get("threshold"))
        return HandlerResult(
            data={"operation": "video.shot.detect", "shots": shots}, outputs=[]
        )
