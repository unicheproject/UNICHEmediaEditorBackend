"""Deterministic image capability handlers (ImageMagick)."""

from __future__ import annotations

from app.capabilities.context import HandlerResult, JobContext, OutputFile
from app.capabilities.handlers._util import (
    input_ext,
    input_stem,
    require,
    require_input,
)
from app.capabilities.handlers.base import LocalToolHandler
from app.models.enums import MediaType
from app.tools import imagemagick

_FORMAT_EXT = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp", "tiff": "tiff"}


def _image_out(ctx: JobContext, suffix: str, ext: str | None = None) -> OutputFile:
    ext = ext or input_ext(ctx)
    filename = f"{input_stem(ctx)}_{suffix}.{ext}"
    return OutputFile(path=ctx.out_path(filename), filename=filename, media_type=MediaType.image)


class ImageResizeHandler(LocalToolHandler):
    capability_id = "image.resize"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _image_out(ctx, "resize")
        await imagemagick.resize(
            src, str(out.path), require(ctx.params, "width"), require(ctx.params, "height")
        )
        return HandlerResult(data={"operation": "image.resize"}, outputs=[out])


class ImageCropHandler(LocalToolHandler):
    capability_id = "image.crop"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _image_out(ctx, "crop")
        await imagemagick.crop(
            src, str(out.path),
            require(ctx.params, "x"), require(ctx.params, "y"),
            require(ctx.params, "width"), require(ctx.params, "height"),
        )
        return HandlerResult(data={"operation": "image.crop"}, outputs=[out])


class ImageFormatHandler(LocalToolHandler):
    capability_id = "image.format"

    async def process(self, ctx: JobContext) -> HandlerResult:
        from app.core.errors import ValidationError

        src = require_input(ctx)
        fmt = str(require(ctx.params, "format")).lstrip(".").lower()
        if fmt not in _FORMAT_EXT:
            raise ValidationError(f"Unsupported target format '{fmt}'")
        out = _image_out(ctx, "convert", ext=_FORMAT_EXT[fmt])
        await imagemagick.convert_format(src, str(out.path))
        return HandlerResult(data={"operation": "image.format", "format": fmt}, outputs=[out])


class ImageColourAdjustHandler(LocalToolHandler):
    capability_id = "image.colour.adjust"

    async def process(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        out = _image_out(ctx, "colour")
        await imagemagick.colour_adjust(
            src, str(out.path),
            brightness=ctx.params.get("brightness", 100),
            saturation=ctx.params.get("saturation", 100),
            contrast=ctx.params.get("contrast", 0),
        )
        return HandlerResult(data={"operation": "image.colour.adjust"}, outputs=[out])
