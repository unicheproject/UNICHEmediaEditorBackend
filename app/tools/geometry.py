"""Tool-neutral geometry checks shared by FFmpeg and ImageMagick wrappers."""

from __future__ import annotations

from app.core.errors import ValidationError

# EXIF orientations (ImageMagick names) that turn the stored image a quarter
# turn when displayed, swapping width and height.
QUARTER_TURN_ORIENTATIONS = frozenset({"LeftTop", "RightTop", "RightBottom", "LeftBottom"})


def is_quarter_turn(rotation_degrees: object) -> bool:
    """True for a video display rotation of ±90/±270 degrees."""
    try:
        return round(float(str(rotation_degrees))) % 180 == 90
    except ValueError:
        return False


def check_crop_rectangle(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    frame_width: int,
    frame_height: int,
    media: str = "video",
) -> None:
    """Reject a crop rectangle that does not lie fully inside the displayed frame.

    FFmpeg's crop filter would otherwise silently move the rectangle inward,
    and ImageMagick would silently clip it (or return a 1×1 image when it lies
    entirely outside), producing a different region than requested.
    """
    if x < 0 or y < 0:
        raise ValidationError("Crop offsets must not be negative")
    if x + width > frame_width or y + height > frame_height:
        raise ValidationError(
            f"Crop rectangle must fit inside the {media} ({frame_width}×{frame_height}): "
            f"x + width ≤ {frame_width} and y + height ≤ {frame_height}"
        )
