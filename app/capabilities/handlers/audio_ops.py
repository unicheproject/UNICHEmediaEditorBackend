"""Deterministic audio capability handlers (FFmpeg)."""

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
from app.tools import ffmpeg


def _audio_out(ctx: JobContext, suffix: str, ext: str | None = None) -> OutputFile:
    ext = ext or input_ext(ctx)
    filename = f"{input_stem(ctx)}_{suffix}.{ext}"
    return OutputFile(path=ctx.out_path(filename), filename=filename, media_type=MediaType.audio)


class AudioTrimHandler(LocalToolHandler):
    capability_id = "audio.trim"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _audio_out(ctx, "trim")
        await ffmpeg.audio_trim(
            src, str(out.path), require(ctx.params, "start"), require(ctx.params, "end")
        )
        return HandlerResult(data={"operation": "audio.trim"}, outputs=[out])


class AudioConcatHandler(LocalToolHandler):
    capability_id = "audio.concat"

    async def process(self, ctx: JobContext) -> HandlerResult:
        if len(ctx.input_paths) < 2:
            raise ValidationError("audio.concat needs at least 2 input assets (input.asset_ids)")
        ext = input_ext(ctx)
        filename = f"{input_stem(ctx)}_concat.{ext}"
        out = OutputFile(path=ctx.out_path(filename), filename=filename, media_type=MediaType.audio)
        await ffmpeg.audio_concat(ctx.input_paths, str(out.path))
        return HandlerResult(
            data={"operation": "audio.concat", "inputs": len(ctx.input_paths)},
            outputs=[out],
        )


class AudioGainHandler(LocalToolHandler):
    capability_id = "audio.gain"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _audio_out(ctx, "gain")
        await ffmpeg.audio_gain(src, str(out.path), require(ctx.params, "gain_db"))
        return HandlerResult(data={"operation": "audio.gain"}, outputs=[out])


class AudioNormalizeHandler(LocalToolHandler):
    capability_id = "audio.normalize"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _audio_out(ctx, "normalized")
        target = ctx.params.get("target_i", -16)
        await ffmpeg.audio_normalize(src, str(out.path), target)
        return HandlerResult(
            data={"operation": "audio.normalize", "target_i": target}, outputs=[out]
        )


class AudioFadeHandler(LocalToolHandler):
    capability_id = "audio.fade"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _audio_out(ctx, "fade")
        await ffmpeg.audio_fade(
            src, str(out.path),
            ctx.params.get("fade_in", 0),
            ctx.params.get("fade_out", 0),
        )
        return HandlerResult(data={"operation": "audio.fade"}, outputs=[out])


class AudioDenoiseHandler(LocalToolHandler):
    capability_id = "audio.denoise"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _audio_out(ctx, "denoise")
        await ffmpeg.audio_denoise(src, str(out.path), ctx.params.get("strength"))
        return HandlerResult(data={"operation": "audio.denoise"}, outputs=[out])


class AudioTranscodeHandler(LocalToolHandler):
    capability_id = "audio.transcode"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        fmt = str(require(ctx.params, "format")).lstrip(".").lower()
        out = _audio_out(ctx, "transcode", ext=fmt)
        await ffmpeg.audio_transcode(src, str(out.path), ctx.params.get("codec"))
        return HandlerResult(data={"operation": "audio.transcode", "format": fmt}, outputs=[out])
