"""Deterministic audio capability handlers (FFmpeg)."""

from __future__ import annotations

from app.capabilities.context import HandlerResult, JobContext, OutputFile
from app.capabilities.handlers._util import (
    input_ext,
    input_stem,
    num_tag,
    require,
    require_input,
    trim_tag,
)
from app.capabilities.handlers.base import LocalToolHandler
from app.capabilities.media_options import AUDIO_TRANSCODE_OPTIONS
from app.core.errors import ValidationError
from app.models.enums import MediaType
from app.tools import ffmpeg


def _audio_out(ctx: JobContext, suffix: str, ext: str | None = None) -> OutputFile:
    ext = ext or input_ext(ctx)
    filename = f"{input_stem(ctx)}_{suffix}.{ext}"
    return OutputFile(path=ctx.out_path(filename), filename=filename, media_type=MediaType.audio)


def _video_out(ctx: JobContext, suffix: str) -> OutputFile:
    ext = input_ext(ctx)
    filename = f"{input_stem(ctx)}_{suffix}.{ext}"
    return OutputFile(path=ctx.out_path(filename), filename=filename, media_type=MediaType.video)


def _fade_suffix(fade_in: object, fade_out: object) -> str:
    parts = [
        f"in{num_tag(fade_in)}" if fade_in else "",
        f"out{num_tag(fade_out)}" if fade_out else "",
    ]
    return "fade_" + "-".join(part for part in parts if part) if any(parts) else "fade"


def _video_input(ctx: JobContext) -> bool:
    return ctx.input_asset_meta.get("media_type") == MediaType.video.value


class AudioTrimHandler(LocalToolHandler):
    capability_id = "audio.trim"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _audio_out(ctx, trim_tag(require(ctx.params, "start"), ctx.params.get("end")))
        await ffmpeg.audio_trim(
            src, str(out.path), require(ctx.params, "start"), ctx.params.get("end")
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
        gain_db = require(ctx.params, "gain_db")
        suffix = f"gain_{num_tag(gain_db)}dB"
        if _video_input(ctx):
            out = _video_out(ctx, suffix)
            await ffmpeg.video_audio_gain(src, str(out.path), gain_db)
            return HandlerResult(data={"operation": "audio.gain"}, outputs=[out])
        out = _audio_out(ctx, suffix)
        await ffmpeg.audio_gain(src, str(out.path), gain_db)
        return HandlerResult(data={"operation": "audio.gain"}, outputs=[out])


class AudioNormalizeHandler(LocalToolHandler):
    capability_id = "audio.normalize"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        target = ctx.params.get("target_i", -16)
        if _video_input(ctx):
            out = _video_out(ctx, f"normalized_{num_tag(target)}LUFS")
            await ffmpeg.video_audio_normalize(src, str(out.path), target)
            return HandlerResult(
                data={"operation": "audio.normalize", "target_i": target}, outputs=[out]
            )
        out = _audio_out(ctx, f"normalized_{num_tag(target)}LUFS")
        await ffmpeg.audio_normalize(src, str(out.path), target)
        return HandlerResult(
            data={"operation": "audio.normalize", "target_i": target}, outputs=[out]
        )


class AudioFadeHandler(LocalToolHandler):
    capability_id = "audio.fade"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        fade_in = ctx.params.get("fade_in", 0)
        fade_out = ctx.params.get("fade_out", 0)
        if _video_input(ctx):
            out = _video_out(ctx, _fade_suffix(fade_in, fade_out))
            await ffmpeg.video_audio_fade(src, str(out.path), fade_in, fade_out)
            return HandlerResult(data={"operation": "audio.fade"}, outputs=[out])
        out = _audio_out(ctx, _fade_suffix(fade_in, fade_out))
        await ffmpeg.audio_fade(
            src, str(out.path), fade_in, fade_out,
        )
        return HandlerResult(data={"operation": "audio.fade"}, outputs=[out])


class AudioDenoiseHandler(LocalToolHandler):
    capability_id = "audio.denoise"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        strength = ctx.params.get("strength")
        suffix = f"denoise_{num_tag(1.0 if strength is None else strength)}"
        if _video_input(ctx):
            out = _video_out(ctx, suffix)
            await ffmpeg.video_audio_denoise(src, str(out.path), strength)
            return HandlerResult(data={"operation": "audio.denoise"}, outputs=[out])
        out = _audio_out(ctx, suffix)
        await ffmpeg.audio_denoise(src, str(out.path), strength)
        return HandlerResult(data={"operation": "audio.denoise"}, outputs=[out])


class VideoAudioExtractHandler(LocalToolHandler):
    capability_id = "video.audio.extract"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        filename = f"{input_stem(ctx)}_audio.m4a"
        out = OutputFile(
            path=ctx.out_path(filename), filename=filename, media_type=MediaType.audio
        )
        await ffmpeg.video_audio_extract(src, str(out.path))
        return HandlerResult(
            data={"operation": "video.audio.extract", "format": "m4a", "codec": "aac"},
            outputs=[out],
        )


class AudioTranscodeHandler(LocalToolHandler):
    capability_id = "audio.transcode"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        fmt = require(ctx.params, "format")
        codec = ctx.params.get("codec")
        if not isinstance(fmt, str) or fmt not in AUDIO_TRANSCODE_OPTIONS:
            raise ValidationError("Unsupported audio output format")
        if "codec" in ctx.params and (
            not isinstance(codec, str) or codec not in AUDIO_TRANSCODE_OPTIONS[fmt]["codec"]
        ):
            raise ValidationError("Unsupported audio codec for the selected output format")
        out = _audio_out(ctx, "transcode", ext=fmt)
        await ffmpeg.audio_transcode(src, str(out.path), codec)
        return HandlerResult(data={"operation": "audio.transcode", "format": fmt}, outputs=[out])
