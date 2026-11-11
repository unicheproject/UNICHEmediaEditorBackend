"""ImageMagick wrappers for deterministic image operations (the `convert` CLI)."""

from __future__ import annotations

import re
from contextlib import suppress
from pathlib import Path

from app.capabilities.media_options import (
    DEFAULT_TEXT_FONT,
    LOWERTHIRD_CAPTION_SIZE,
    LOWERTHIRD_TEXT_SIZE,
    TEXT_ALIGNS,
    TEXT_COLOUR_PATTERN,
    TEXT_FONT_FILES,
    TITLECARD_FONT_SIZE,
    TITLECARD_MAX_PIXELS,
    TITLECARD_MAX_SIDE,
    VERTICAL_ALIGNS,
)
from app.core.errors import ToolError, ValidationError
from app.tools import runner
from app.tools.geometry import QUARTER_TURN_ORIENTATIONS, check_crop_rectangle

CONVERT = "convert"
IDENTIFY = "identify"

# Bundled in the Docker image via fonts-dejavu-core. A font is always passed:
# without -font, convert falls back to the policy's missing font and aborts.
DEFAULT_FONT = TEXT_FONT_FILES[DEFAULT_TEXT_FONT]
_COLOUR = re.compile(TEXT_COLOUR_PATTERN)
# (vertical, horizontal) -> caption gravity; the default keeps "center".
_TITLE_GRAVITY = {
    ("top", "left"): "NorthWest", ("top", "center"): "North", ("top", "right"): "NorthEast",
    ("middle", "left"): "West", ("middle", "center"): "center", ("middle", "right"): "East",
    ("bottom", "left"): "SouthWest", ("bottom", "center"): "South",
    ("bottom", "right"): "SouthEast",
}
# Lower-third layout (640x160 banner): the name's box starts at +20+16 and
# the caption at +20+72 for the default 36-point name.
_LT_X, _LT_TOP, _LT_TEXT, _LT_CAPTION = 20, 16, 36, 22
_MEASURE_TIMEOUT = 30.0


def im_text(text: str) -> str:
    """User text as literal ImageMagick text (caption:, label:, -annotate).

    ImageMagick expands %-escapes (e.g. %w, %[fx:...]), backslash escapes and a
    leading @ (read text from a file) in these arguments. Doubling % and \\ and
    escaping a leading @ makes them print as typed; real newlines still break
    lines. Apply exactly once, where the argument is built.
    """
    escaped = text.replace("\\", "\\\\").replace("%", "%%")
    return "\\" + escaped if escaped.startswith("@") else escaped


def _colour(value: object, name: str) -> str:
    if not isinstance(value, str) or not _COLOUR.fullmatch(value):
        raise ValidationError(f"{name} must be #RGB, #RRGGBB or a colour name")
    return value


def _font_file(font: object) -> str:
    if font is None:
        return DEFAULT_FONT
    if not isinstance(font, str) or font not in TEXT_FONT_FILES:
        raise ValidationError("Unsupported font")
    return TEXT_FONT_FILES[font]


def _size(value: object, bounds: tuple[int, int], name: str) -> int:
    low, high = bounds
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValidationError(f"{name} must be an integer from {low} to {high}")
    return value


async def _label_box(text: str, font: str, size: int) -> tuple[int, int]:
    """Width and height of the text box -annotate draws (font metrics)."""
    out = await runner.run_stdout(
        [CONVERT, "-font", font, "-pointsize", str(size), f"label:{im_text(text)}",
         "-format", "%w %h", "info:"],
        timeout=_MEASURE_TIMEOUT,
    )
    try:
        width, height = (int(part) for part in out.split())
    except ValueError:
        raise ToolError("Text could not be measured") from None
    return width, height


def _int(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"Parameter '{name}' must be an integer, got {value!r}")
    return str(value)


async def resize(
    src: str,
    dst: str,
    width: int | None = None,
    height: int | None = None,
    *,
    fit: str = "stretch",
    quality: int | None = None,
) -> None:
    """Resize the first frame as displayed; one side keeps the aspect ratio.

    With both sides, fit chooses: stretch to exactly WxH ("WxH!"), contain
    inside WxH ("WxH"), or cover WxH and crop the centre ("WxH^" + extent).
    """
    if width is None and height is None:
        raise ValidationError("Resize needs a width, a height, or both")
    if fit not in ("stretch", "contain", "cover"):
        raise ValidationError(f"Unsupported resize fit '{fit}'")
    both = width is not None and height is not None
    ops: list[str]
    if both:
        box = f"{_int(width, 'width')}x{_int(height, 'height')}"
        if fit == "stretch":
            ops = ["-resize", f"{box}!"]
        elif fit == "contain":
            ops = ["-resize", box]
        else:
            ops = ["-resize", f"{box}^", "-gravity", "center", "-extent", box, "+repage"]
    elif width is not None:
        ops = ["-resize", f"{_int(width, 'width')}x"]
    else:
        ops = ["-resize", f"x{_int(height, 'height')}"]
    if quality is not None:
        ops += ["-quality", _int(quality, "quality")]
    output = Path(dst)
    validated = False
    try:
        # [0]: multi-page inputs would otherwise write dst-0, dst-1, ... and no dst.
        await runner.run([CONVERT, f"{src}[0]", "-auto-orient", *ops, dst])
        if not output.is_file() or output.stat().st_size == 0:
            raise ToolError("Resize output is missing or empty")
        out_width, out_height = await display_size(dst)
        if both and fit == "contain":
            assert width is not None and height is not None
            fits = out_width <= width and out_height <= height
            if not fits or (out_width != width and out_height != height):
                raise ToolError("Resize output does not match the requested size")
        elif (width is not None and out_width != width) or (
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
    if orientation.strip() in QUARTER_TURN_ORIENTATIONS:
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


# Encoder settings used when the caller gives no quality, stated rather than
# left to ImageMagick (whose WebP default is 75 and JPEG default follows the input).
DEFAULT_IMAGE_QUALITY = {"jpeg": 92, "webp": 90}
LOSSY_IMAGE_FORMATS = frozenset(DEFAULT_IMAGE_QUALITY)
LOSSY_EXTENSIONS = frozenset({"jpg", "jpeg", "webp"})


async def convert_format(
    src: str,
    dst: str,
    *,
    fmt: str,
    quality: int | None = None,
    lossless: bool = False,
    flatten: bool = False,
) -> dict[str, object]:
    """Convert the first frame, as displayed, and report the encoding used.

    JPEG cannot hold transparency, so it is always flattened onto white; TIFF
    is written with lossless LZW compression whatever the source used.
    """
    args = [CONVERT, f"{src}[0]", "-auto-orient"]
    if flatten or fmt == "jpeg":
        args += ["-background", "white", "-alpha", "remove", "-alpha", "off"]
    used: dict[str, object] = {"lossless": fmt not in LOSSY_IMAGE_FORMATS or lossless}
    if fmt in LOSSY_IMAGE_FORMATS and not lossless:
        value = DEFAULT_IMAGE_QUALITY[fmt] if quality is None else quality
        args += ["-quality", _int(value, "quality")]
        used["quality"] = value
    elif fmt == "webp":
        args += ["-define", "webp:lossless=true", "-quality", "100"]
    elif fmt == "tiff":
        args += ["-compress", "LZW"]
    output = Path(dst)
    validated = False
    try:
        # [0]: multi-page inputs would otherwise write dst-0, dst-1, ... and no dst.
        await runner.run([*args, dst])
        if not output.is_file() or output.stat().st_size == 0:
            raise ToolError("Format conversion output is missing or empty")
        await display_size(dst)
        validated = True
    finally:
        if not validated:
            with suppress(OSError):
                output.unlink()
    return used


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
    font: str | None = None,
    font_size: int | None = None,
    text_align: str | None = None,
    vertical_align: str | None = None,
) -> None:
    """Render word-wrapped text onto a solid background PNG.

    Without font_size, `caption:` sizes the text to fill the canvas (unchanged
    behaviour). With it, that exact size is used and text that would not fit
    the canvas is rejected instead of being clipped.
    """
    w, h = int(_int(width, "width")), int(_int(height, "height"))
    if w > TITLECARD_MAX_SIDE or h > TITLECARD_MAX_SIDE or w * h > TITLECARD_MAX_PIXELS:
        raise ValidationError("Title card is larger than 3840 pixels or 3840x2160 in area")
    horizontal = "center" if text_align is None else text_align
    vertical = "middle" if vertical_align is None else vertical_align
    if horizontal not in TEXT_ALIGNS or vertical not in VERTICAL_ALIGNS:
        raise ValidationError("Unsupported title alignment")
    font_file = _font_file(font)
    args = [CONVERT, "-size", f"{w}x{h}"]
    args += ["-background", _colour(background, "Background colour"),
             "-fill", _colour(foreground, "Text colour"),
             "-gravity", _TITLE_GRAVITY[(vertical, horizontal)], "-font", font_file]
    if font_size is not None:
        size = _size(font_size, TITLECARD_FONT_SIZE, "Font size")
        # caption: wraps at spaces but cuts a word wider than the card (measured),
        # so every word is measured first, in a single call.
        words = sorted(set(text.split()))
        if words:
            widths = await runner.run_stdout(
                [CONVERT, "-font", font_file, "-pointsize", str(size),
                 *(f"label:{im_text(word)}" for word in words),
                 "-format", "%w\n", "info:"],
                timeout=_MEASURE_TIMEOUT,
            )
            measured = widths.split()
            if len(measured) != len(words) or not all(v.isdigit() for v in measured):
                raise ToolError("Title text could not be measured")
            if max(int(v) for v in measured) > w:
                raise ValidationError(
                    "A word in the title is too wide for the card at this font size; "
                    "use a smaller size"
                )
        needed = await runner.run_stdout(
            [CONVERT, "-size", f"{w}x", "-font", font_file, "-pointsize", str(size),
             f"caption:{im_text(text)}", "-format", "%h", "info:"],
            timeout=_MEASURE_TIMEOUT,
        )
        if not needed.strip().isdigit():
            raise ToolError("Title text could not be measured")
        if int(needed) > h:
            raise ValidationError(
                "The title does not fit on the card at this font size; use a smaller "
                "size or shorter text"
            )
        args += ["-pointsize", str(size)]
    args += [f"caption:{im_text(text)}", dst]
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
    font: str | None = None,
    text_size: int | None = None,
    caption_size: int | None = None,
) -> None:
    """Render a small solid-background name/caption box PNG (not full-frame) --
    meant to be overlaid onto an existing video, not to fill it.

    Text that would run past the banner, or a name that would overlap the
    caption, is rejected instead of being clipped.
    """
    w, h = int(_int(width, "width")), int(_int(height, "height"))
    font_file = _font_file(font)
    name_size = _LT_TEXT if text_size is None else _size(
        text_size, LOWERTHIRD_TEXT_SIZE, "Text size")
    note_size = _LT_CAPTION if caption_size is None else _size(
        caption_size, LOWERTHIRD_CAPTION_SIZE, "Caption size")
    # 72 for the default 36-point name; scales with a chosen name size.
    caption_top = 72 if text_size is None else _LT_TOP + round(name_size * 14 / 9)
    args = [
        CONVERT,
        "-size", f"{w}x{h}",
        f"xc:{_colour(background, 'Background colour')}", "-gravity", "northwest",
        "-fill", _colour(foreground, "Text colour"), "-font", font_file,
    ]
    name_w, name_h = await _label_box(text, font_file, name_size)
    if _LT_X + name_w > w or _LT_TOP + name_h > (caption_top if caption else h):
        raise ValidationError(
            "The lower-third name does not fit in the banner; shorten it or use a "
            "smaller text size"
        )
    args += ["-pointsize", str(name_size), "-annotate", f"+{_LT_X}+{_LT_TOP}", im_text(text)]
    if caption:
        note_w, note_h = await _label_box(caption, font_file, note_size)
        if _LT_X + note_w > w or caption_top + note_h > h:
            raise ValidationError(
                "The lower-third caption does not fit in the banner; shorten it or use "
                "a smaller caption size"
            )
        args += ["-pointsize", str(note_size), "-annotate", f"+{_LT_X}+{caption_top}",
                 im_text(caption)]
    args.append(dst)
    await runner.run(args)
