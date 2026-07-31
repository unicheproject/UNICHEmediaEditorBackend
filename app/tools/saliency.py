"""Focus-point detection for image.kenburns's saliency-aware pan/zoom.

No extra dependency: face detection uses OpenCV's bundled Haar cascade
(cv2.data.haarcascades), and the no-face fallback scores "visual density" via
a Laplacian edge-energy map -- classical CV techniques, not the cv2.saliency
module (which needs opencv-contrib-python, conflicting with the opencv-python
already required by video.shot.detect/image.upscale's dependencies).

Detection is synchronous/CPU-bound, so it's offloaded to a thread via
asyncio.to_thread, matching app/tools/pyscenedetect.py's convention.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.errors import ToolError

_FACE_CASCADE_NAME = "haarcascade_frontalface_default.xml"


def _face_focus(gray: Any, cascade: Any) -> tuple[float, float] | None:
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # The largest detected face is almost certainly the intended subject.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    height, width = gray.shape
    return (x + w / 2) / width, (y + h / 2) / height


def _energy_focus(gray: Any) -> tuple[float, float]:
    """Fallback: weighted centroid of edge-energy (Laplacian magnitude).

    No dominant face -- treat the highest-detail region as the focal point, a
    classical, fully deterministic proxy for visual saliency.
    """
    import cv2
    import numpy as np

    energy = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    total = energy.sum()
    height, width = gray.shape
    if total <= 0:
        return 0.5, 0.5
    ys, xs = np.indices(energy.shape)
    cx = float((xs * energy).sum() / total)
    cy = float((ys * energy).sum() / total)
    return cx / width, cy / height


def _find_focus_sync(path: str) -> tuple[float, float]:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ToolError(f"image.kenburns could not read input image: {path}")

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + _FACE_CASCADE_NAME)  # type: ignore[attr-defined]
    focus = None if cascade.empty() else _face_focus(img, cascade)
    if focus is None:
        focus = _energy_focus(img)
    return focus


async def find_focus_point(src: str) -> tuple[float, float]:
    """Return the (x, y) focus point as fractions (0.0-1.0) of image width/height."""
    return await asyncio.to_thread(_find_focus_sync, src)
