"""Composition capability handlers (slideshow, title card, subtitle, mix, compose).

These build on FFmpeg/ImageMagick like the other tool handlers but assemble
clips/audio/subtitles. Their outputs are derived video assets that can be
chained (e.g. title card + slideshow -> compose).
"""

from __future__ import annotations

from app.capabilities.context import HandlerResult, JobContext, OutputFile
from app.capabilities.handlers._util import input_stem, require, require_input
from app.capabilities.handlers.base import LocalToolHandler
from app.core.errors import ValidationError
from app.models.enums import MediaType
from app.tools import ffmpeg, imagemagick

DEFAULT_W, DEFAULT_H = 1920, 1080


def _wh(ctx: JobContext) -> tuple[int, int]:
    return ctx.params.get("width", DEFAULT_W), ctx.params.get("height", DEFAULT_H)


def _video_out(ctx: JobContext, name: str) -> OutputFile:
    filename = f"{name}.mp4"
    return OutputFile(path=ctx.out_path(filename), filename=filename, media_type=MediaType.video)


def _named(ctx: JobContext, key: str) -> str:
    path = ctx.named_input_paths.get(key)
    if not path:
        raise ValidationError(f"Missing required input '{key}'")
    return path


class ImageSlideshowHandler(LocalToolHandler):
    capability_id = "image.slideshow"

    async def process(self, ctx: JobContext) -> HandlerResult:
        if not ctx.input_paths:
            raise ValidationError("image.slideshow needs input images (asset_id / input.asset_ids)")
        w, h = _wh(ctx)
        out = _video_out(ctx, "slideshow")
        await ffmpeg.slideshow(
            ctx.input_paths, str(out.path),
            seconds_per_image=ctx.params.get("seconds_per_image", 4),
            width=w, height=h,
        )
        return HandlerResult(
            data={"operation": "image.slideshow", "images": len(ctx.input_paths)},
            outputs=[out],
        )


class MediaTitlecardHandler(LocalToolHandler):
    capability_id = "media.titlecard"

    async def process(self, ctx: JobContext) -> HandlerResult:
        text = require(ctx.params, "text")
        w, h = _wh(ctx)
        duration = ctx.params.get("duration", 5)
        png = ctx.out_path("titlecard.png")
        await imagemagick.render_text_card(
            str(png), str(text), width=w, height=h,
            background=ctx.params.get("background", "#101418"),
            foreground=ctx.params.get("foreground", "#ffffff"),
        )
        out = _video_out(ctx, "titlecard")
        await ffmpeg.still_to_clip(str(png), str(out.path), duration=duration, width=w, height=h)
        return HandlerResult(data={"operation": "media.titlecard"}, outputs=[out])


class VideoSubtitleEmbedHandler(LocalToolHandler):
    capability_id = "video.subtitle.embed"

    async def process(self, ctx: JobContext) -> HandlerResult:
        video = require_input(ctx)
        subs = _named(ctx, "subtitle_asset_id")
        mode = ctx.params.get("mode", "soft")
        if mode not in ("soft", "burn"):
            raise ValidationError("mode must be 'soft' or 'burn'")
        out = OutputFile(
            path=ctx.out_path(f"{input_stem(ctx)}_subtitled.mp4"),
            filename=f"{input_stem(ctx)}_subtitled.mp4",
            media_type=MediaType.video,
        )
        await ffmpeg.subtitle_embed(video, subs, str(out.path), mode=mode)
        return HandlerResult(
            data={"operation": "video.subtitle.embed", "mode": mode}, outputs=[out]
        )


class AudioMixHandler(LocalToolHandler):
    capability_id = "audio.mix"

    async def process(self, ctx: JobContext) -> HandlerResult:
        media = require_input(ctx)
        music = _named(ctx, "music_asset_id")
        mode = ctx.params.get("mode", "mix")
        # Output keeps the primary's media kind (video stays video, audio stays audio).
        is_video = ctx.input_asset_meta.get("media_type") == "video"
        ext = "mp4" if is_video else "m4a"
        mtype = MediaType.video if is_video else MediaType.audio
        filename = f"{input_stem(ctx)}_mixed.{ext}"
        out = OutputFile(path=ctx.out_path(filename), filename=filename, media_type=mtype)
        await ffmpeg.audio_mix(
            media, music, str(out.path),
            music_volume=ctx.params.get("music_volume", 0.3), mode=mode,
        )
        return HandlerResult(data={"operation": "audio.mix", "mode": mode}, outputs=[out])


class VideoComposeHandler(LocalToolHandler):
    capability_id = "video.compose"

    async def process(self, ctx: JobContext) -> HandlerResult:
        if len(ctx.input_paths) < 1:
            raise ValidationError("video.compose needs at least one segment (input.asset_ids)")
        assert ctx.work_dir is not None
        w, h = _wh(ctx)
        out = _video_out(ctx, "composition")
        await ffmpeg.compose(
            ctx.input_paths, str(out.path),
            width=w, height=h, work_dir=ctx.work_dir,
            audio_path=ctx.named_input_paths.get("audio_asset_id"),
            audio_mode=ctx.params.get("audio_mode", "mix"),
            music_volume=ctx.params.get("music_volume", 0.3),
            subtitle_path=ctx.named_input_paths.get("subtitle_asset_id"),
        )
        return HandlerResult(
            data={"operation": "video.compose", "segments": len(ctx.input_paths)},
            outputs=[out],
        )
