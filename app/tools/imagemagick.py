"""ImageMagick wrappers for deterministic image operations (the `convert` CLI)."""

from __future__ import annotations

from app.core.errors import ToolError
from app.tools import runner

CONVERT = "convert"


def _int(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"Parameter '{name}' must be an integer, got {value!r}")
    return str(value)


async def resize(src: str, dst: str, width: int, height: int) -> None:
    # "WxH!" forces exact dimensions (ignore aspect ratio).
    geom = f"{_int(width, 'width')}x{_int(height, 'height')}!"
    await runner.run([CONVERT, src, "-resize", geom, dst])


async def crop(src: str, dst: str, x: int, y: int, width: int, height: int) -> None:
    geom = (
        f"{_int(width, 'width')}x{_int(height, 'height')}"
        f"+{_int(x, 'x')}+{_int(y, 'y')}"
    )
    await runner.run([CONVERT, src, "-crop", geom, "+repage", dst])


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
