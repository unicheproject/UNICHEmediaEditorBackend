"""Unit tests for saliency-based focus-point detection (image.kenburns)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from app.tools import saliency

HAVE_CONVERT = shutil.which("convert") is not None
pytestmark = pytest.mark.skipif(not HAVE_CONVERT, reason="needs ImageMagick's convert")


def _run(args: list[str]) -> None:
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())


async def test_find_focus_point_locates_bright_spot(tmp_path) -> None:
    img = tmp_path / "spot.png"
    _run([
        "convert", "-size", "400x300", "xc:#202020",
        "-fill", "white", "-draw", "circle 320,60 340,60",
        str(img),
    ])

    fx, fy = await saliency.find_focus_point(str(img))

    # Spot is at (320, 60) of 400x300 = (0.8, 0.2); allow tolerance for the
    # anti-aliased circle edge shifting the energy centroid slightly.
    assert 0.65 <= fx <= 0.95
    assert 0.05 <= fy <= 0.35


async def test_find_focus_point_defaults_to_center_on_blank_image(tmp_path) -> None:
    img = tmp_path / "blank.png"
    _run(["convert", "-size", "200x200", "xc:#333333", str(img)])

    fx, fy = await saliency.find_focus_point(str(img))

    assert fx == pytest.approx(0.5)
    assert fy == pytest.approx(0.5)
