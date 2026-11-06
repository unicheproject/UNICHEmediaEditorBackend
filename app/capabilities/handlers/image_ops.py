"""Deterministic image capability handlers (ImageMagick, Real-ESRGAN)."""

from __future__ import annotations

from app.capabilities.context import HandlerResult, JobContext, OutputFile
from app.capabilities.handlers._util import (
    input_ext,
    input_stem,
    num_tag,
    require,
    require_input,
)
from app.capabilities.handlers.base import LocalToolHandler
from app.core.errors import ValidationError
from app.models.enums import MediaType
from app.tools import imagemagick, realesrgan

_UPSCALE_EXTS = {"jpg", "jpeg", "png", "webp"}

_FORMAT_EXT = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp", "tiff": "tiff"}


def _image_out(ctx: JobContext, suffix: str, ext: str | None = None) -> OutputFile:
    ext = ext or input_ext(ctx)
    filename = f"{input_stem(ctx)}_{suffix}.{ext}"
    return OutputFile(path=ctx.out_path(filename), filename=filename, media_type=MediaType.image)


class ImageResizeHandler(LocalToolHandler):
    capability_id = "image.resize"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        work = _image_out(ctx, "resize")
        await imagemagick.resize(
            src, str(work.path), ctx.params.get("width"), ctx.params.get("height")
        )
        width, height = await imagemagick.display_size(str(work.path))
        out = OutputFile(
            path=work.path, filename=f"{input_stem(ctx)}_resize_{width}x{height}.{input_ext(ctx)}",
            media_type=MediaType.image,
        )
        return HandlerResult(data={"operation": "image.resize"}, outputs=[out])


class ImageCropHandler(LocalToolHandler):
    capability_id = "image.crop"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _image_out(
            ctx, f"crop_{require(ctx.params, 'width')}x{require(ctx.params, 'height')}"
        )
        await imagemagick.crop(
            src, str(out.path),
            require(ctx.params, "x"), require(ctx.params, "y"),
            require(ctx.params, "width"), require(ctx.params, "height"),
        )
        return HandlerResult(data={"operation": "image.crop"}, outputs=[out])


class ImageFormatHandler(LocalToolHandler):
    capability_id = "image.format"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        fmt = str(require(ctx.params, "format")).lstrip(".").lower()
        if fmt == "jpg":  # accepted from older clients; same format as jpeg
            fmt = "jpeg"
        if fmt not in _FORMAT_EXT:
            raise ValidationError(f"Unsupported target format '{fmt}'")
        quality = ctx.params.get("quality")
        lossless = bool(ctx.params.get("lossless", False))
        if quality is not None and fmt not in imagemagick.LOSSY_IMAGE_FORMATS:
            raise ValidationError(f"Quality applies only to JPEG and WebP, not '{fmt}'")
        if lossless and fmt != "webp":
            raise ValidationError("Lossless applies only to WebP")
        if lossless and quality is not None:
            raise ValidationError("Choose either a quality or lossless WebP, not both")
        variant = "lossless" if lossless else (f"q{quality}" if quality is not None else "")
        suffix = f"convert_{variant}" if variant else "convert"
        out = _image_out(ctx, suffix, ext=_FORMAT_EXT[fmt])
        used = await imagemagick.convert_format(
            src, str(out.path), fmt=fmt, quality=quality, lossless=lossless,
            flatten=bool(ctx.params.get("flatten_transparency", False)),
        )
        return HandlerResult(
            data={"operation": "image.format", "format": fmt, **used}, outputs=[out]
        )


class ImageUpscaleHandler(LocalToolHandler):
    capability_id = "image.upscale"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        ext = input_ext(ctx)
        if ext not in _UPSCALE_EXTS:
            raise ValidationError(f"image.upscale does not support '.{ext}' input images")
        scale = ctx.params.get("scale")
        out = _image_out(ctx, f"upscaled_x{scale or 4}")
        await realesrgan.upscale(src, str(out.path), scale)
        return HandlerResult(
            data={"operation": "image.upscale", "scale": scale or 4}, outputs=[out]
        )


class ImageColourAdjustHandler(LocalToolHandler):
    capability_id = "image.colour.adjust"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        brightness = ctx.params.get("brightness", 100)
        saturation = ctx.params.get("saturation", 100)
        contrast = ctx.params.get("contrast", 0)
        out = _image_out(
            ctx,
            f"colour_b{num_tag(brightness)}-c{num_tag(contrast)}-s{num_tag(saturation)}",
        )
        await imagemagick.colour_adjust(
            src, str(out.path),
            brightness=brightness, saturation=saturation, contrast=contrast,
        )
        return HandlerResult(data={"operation": "image.colour.adjust"}, outputs=[out])
