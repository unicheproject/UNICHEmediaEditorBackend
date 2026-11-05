"""ImageMagick wrappers for deterministic image operations (the `convert` CLI)."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from app.core.errors import ToolError, ValidationError
from app.tools import runner
from app.tools.geometry import check_crop_rectangle

CONVERT = "convert"
IDENTIFY = "identify"

# Bundled in the Docker image via fonts-dejavu-core.
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _int(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"Parameter '{name}' must be an integer, got {value!r}")
    return str(value)


async def resize(
    src: str, dst: str, width: int | None = None, height: int | None = None
) -> None:
    """Resize the first frame as displayed; one side keeps the aspect ratio.

    "WxH!" forces exact dimensions; "Wx" / "xH" preserve the aspect ratio.
    """
    if width is None and height is None:
        raise ValidationError("Resize needs a width, a height, or both")
    if width is not None and height is not None:
        geom = f"{_int(width, 'width')}x{_int(height, 'height')}!"
    elif width is not None:
        geom = f"{_int(width, 'width')}x"
    else:
        geom = f"x{_int(height, 'height')}"
    output = Path(dst)
    validated = False
    try:
        # [0]: multi-page inputs would otherwise write dst-0, dst-1, ... and no dst.
        await runner.run([CONVERT, f"{src}[0]", "-auto-orient", "-resize", geom, dst])
        if not output.is_file() or output.stat().st_size == 0:
            raise ToolError("Resize output is missing or empty")
        out_width, out_height = await display_size(dst)
        if (width is not None and out_width != width) or (
            height is not None and out_height != height
        ):
            raise ToolError("Resize output does not match the requested size")
        validated = True
    finally:
        if not validated:
            with suppress(OSError):
                output.unlink()


async def thumbnail(src: str, dst: str, max_size: int = 512) -> None:
    """Render the first image/frame as a browser-safe JPEG thumbnail."""
    geometry = f"{_int(max_size, 'max_size')}x{_int(max_size, 'max_size')}>"
    await runner.run(
        [
            CONVERT,
            f"{src}[0]",
            "-auto-orient",
            "-thumbnail",
            geometry,
            "-colorspace",
            "sRGB",
            "-background",
            "white",
            "-alpha",
            "remove",
            "-alpha",
            "off",
            "-strip",
            "-quality",
            "85",
            dst,
        ]
    )


# EXIF orientations that turn the stored image a quarter turn when displayed.
_QUARTER_TURN_ORIENTATIONS = {"LeftTop", "RightTop", "RightBottom", "LeftBottom"}


async def display_size(src: str) -> tuple[int, int]:
    """First frame's size as displayed (after EXIF orientation), like thumbnails."""
    out = await runner.run_stdout(
        [IDENTIFY, "-ping", "-format", "%w %h %[orientation]\n", f"{src}[0]"]
    )
    try:
        raw_width, raw_height, orientation = out.splitlines()[0].split(" ", 2)
        width, height = int(raw_width), int(raw_height)
    except (IndexError, ValueError) as exc:
        raise ToolError("Cannot read source image dimensions") from exc
    if orientation.strip() in _QUARTER_TURN_ORIENTATIONS:
        return height, width
    return width, height


async def crop(src: str, dst: str, x: int, y: int, width: int, height: int) -> None:
    """Crop the first frame, as displayed, to a rectangle that must fit inside it."""
    geom = (
        f"{_int(width, 'width')}x{_int(height, 'height')}"
        f"+{_int(x, 'x')}+{_int(y, 'y')}"
    )
    frame_width, frame_height = await display_size(src)
    check_crop_rectangle(
        x, y, width, height, frame_width=frame_width, frame_height=frame_height,
        media="image",
    )
    output = Path(dst)
    validated = False
    try:
        # [0]: multi-page inputs would otherwise write dst-0, dst-1, ... and no dst.
        await runner.run([CONVERT, f"{src}[0]", "-auto-orient", "-crop", geom, "+repage", dst])
        if not output.is_file() or output.stat().st_size == 0:
            raise ToolError("Crop output is missing or empty")
        if await display_size(dst) != (width, height):
            raise ToolError("Crop output does not match the requested size")
        validated = True
    finally:
        if not validated:
            with suppress(OSError):
                output.unlink()


async def convert_format(src: str, dst: str) -> None:
    # Output format is inferred from the destination file extension.
    await runner.run([CONVERT, src, dst])


async def colour_adjust(
    src: str,
    dst: str,
    brightness: int = 100,
    saturation: int = 100,
    contrast: int = 0,
) -> None:
    # -modulate brightness,saturation,hue (100 = unchanged).
    # -brightness-contrast contrast component (-100..100, 0 = unchanged).
    modulate = f"{_int(brightness, 'brightness')},{_int(saturation, 'saturation')},100"
    args = [CONVERT, src, "-modulate", modulate]
    if contrast:
        args += ["-brightness-contrast", f"0x{_int(contrast, 'contrast')}"]
    args.append(dst)
    await runner.run(args)


async def render_text_card(
    dst: str,
    text: str,
    *,
    width: int,
    height: int,
    background: str = "#101418",
    foreground: str = "#ffffff",
    font: str | None = DEFAULT_FONT,
) -> None:
    """Render centered, word-wrapped text onto a solid background PNG."""
    # `caption:` auto-wraps and auto-sizes text to the given canvas size.
    args = [CONVERT, "-size", f"{_int(width, 'width')}x{_int(height, 'height')}"]
    args += ["-background", background, "-fill", foreground, "-gravity", "center"]
    if font:
        args += ["-font", font]
    args += [f"caption:{text}", dst]
    await runner.run(args)


async def render_lowerthird(
    dst: str,
    text: str,
    caption: str | None = None,
    *,
    width: int = 640,
    height: int = 160,
    background: str = "#101418",
    foreground: str = "#ffffff",
    font: str | None = DEFAULT_FONT,
) -> None:
    """Render a small solid-background name/caption box PNG (not full-frame) --
    meant to be overlaid onto an existing video, not to fill it."""
    args = [
        CONVERT,
        "-size", f"{_int(width, 'width')}x{_int(height, 'height')}",
        f"xc:{background}", "-gravity", "northwest", "-fill", foreground,
    ]
    if font:
        args += ["-font", font]
    args += ["-pointsize", "36", "-annotate", "+20+16", text]
    if caption:
        args += ["-pointsize", "22", "-annotate", "+20+72", caption]
    args.append(dst)
    await runner.run(args)
