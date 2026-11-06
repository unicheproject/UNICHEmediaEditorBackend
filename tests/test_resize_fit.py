"""Resize fit modes, distortion reporting and quality (devfixes C-1, S-1)."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.capabilities import registry
from app.capabilities.handlers._util import aspect_ratio_change
from app.capabilities.validation import (
    CapabilityInputValidationError,
    validate_capability_input,
)
from app.core.database import async_session_factory
from app.core.errors import ValidationError
from app.services.media_metadata import probe_asset_metadata
from app.tools import ffmpeg

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_CONVERT = shutil.which("convert") is not None


def test_aspect_ratio_change() -> None:
    # The report's case: 2400x1600 (3:2) forced into 1600x1000 is 6.7% wider.
    assert aspect_ratio_change((2400, 1600), (1600, 1000)) == pytest.approx(0.0667, abs=1e-4)
    assert aspect_ratio_change((2400, 1600), (1600, 1067)) == pytest.approx(0, abs=1e-3)
    assert aspect_ratio_change((0, 10), (10, 10)) is None


_ACCEPTED_RESIZE_INPUTS = [
    # Clients that seed the schema default send fit "stretch" with one side.
    {"width": 640, "fit": "stretch"},
    {"height": 480, "fit": "stretch"},
    {"width": 640},
    {"height": 480},
    {"width": 640, "height": 480},
    {"width": 640, "height": 480, "fit": "stretch"},
    {"width": 640, "height": 480, "fit": "contain"},
    {"width": 640, "height": 480, "fit": "cover"},
]
_REJECTED_RESIZE_INPUTS = [
    {},
    {"fit": "stretch"},
    {"fit": "contain"},
    {"width": 640, "height": 480, "fit": "pad"},
    {"width": 640, "fit": "pad"},
    # Contain and cover are defined against a box: one side is not enough.
    {"width": 640, "fit": "contain"},
    {"height": 480, "fit": "contain"},
    {"width": 640, "fit": "cover"},
    {"height": 480, "fit": "cover"},
]


@pytest.mark.parametrize("capability", ["image.resize", "video.resize"])
@pytest.mark.parametrize("params", _ACCEPTED_RESIZE_INPUTS)
def test_resize_contract_accepts(capability: str, params: dict) -> None:
    validate_capability_input(registry.get(capability), params)


@pytest.mark.parametrize("capability", ["image.resize", "video.resize"])
@pytest.mark.parametrize("params", _REJECTED_RESIZE_INPUTS)
def test_resize_contract_rejects(capability: str, params: dict) -> None:
    with pytest.raises(CapabilityInputValidationError):
        validate_capability_input(registry.get(capability), params)


def test_video_scale_filters() -> None:
    assert ffmpeg.resize_scale_filter(640, None) == "scale=640:-2"
    assert ffmpeg.resize_scale_filter(641, 481, "stretch") == "scale=640:480"
    assert ffmpeg.resize_scale_filter(640, 480, "contain") == (
        "scale=640:480:force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    assert ffmpeg.resize_scale_filter(640, 480, "cover").endswith(",crop=640:480")
    # One side given: fit is irrelevant, aspect is kept.
    assert ffmpeg.resize_scale_filter(None, 480, "cover") == "scale=-2:480"
    with pytest.raises(ValidationError):
        ffmpeg.resize_scale_filter(640, 480, "pad")


async def _job(client: AsyncClient, asset_id: str, capability: str, params: dict) -> dict:
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": capability, "asset_id": asset_id, "input": params,
    })
    assert resp.status_code == 201, resp.text
    job: dict = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded", job
    return job["output"]


async def _upload(client: AsyncClient, pid: str, path: Path, mime: str) -> str:
    resp = await client.post(
        f"{API}/projects/{pid}/assets", files={"file": (path.name, path.read_bytes(), mime)}
    )
    assert resp.status_code == 201, resp.text
    asset_id: str = resp.json()["id"]
    async with async_session_factory() as session:
        await probe_asset_metadata(session, uuid.UUID(asset_id))
    return asset_id


def _identify(path: Path, fmt: str) -> str:
    return subprocess.run(["identify", "-format", fmt, f"{path}[0]"], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
async def test_image_fit_modes_and_distortion_report(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "scan.jpg"
    subprocess.run(["convert", "-size", "2400x1600", "gradient:navy-gold", "-quality", "95",
                    str(src)], check=True)
    pid = await make_project("resize-fit-image")
    aid = await _upload(client, pid, src, "image/jpeg")

    stretched = await _job(client, aid, "image.resize", {"width": 1600, "height": 1000})
    assert (stretched["width"], stretched["height"], stretched["fit"]) == (1600, 1000, "stretch")
    assert stretched["aspect_ratio_change"] == pytest.approx(0.0667, abs=1e-3)

    locked = await _job(client, aid, "image.resize", {"width": 1600})
    assert (locked["width"], locked["height"]) == (1600, 1067)
    assert abs(locked["aspect_ratio_change"]) < 0.001

    contain = await _job(client, aid, "image.resize",
                         {"width": 1600, "height": 1000, "fit": "contain"})
    assert (contain["width"], contain["height"]) == (1500, 1000)
    cover = await _job(client, aid, "image.resize",
                       {"width": 1600, "height": 1000, "fit": "cover"})
    assert (cover["width"], cover["height"]) == (1600, 1000)
    assert cover["aspect_ratio_change"] == 0
    assert cover["outputs"][0]["filename"] == "scan_resize_1600x1000_2.jpg"

    low = await _job(client, aid, "image.resize", {"width": 1200, "quality": 40})
    path = tmp_path / "low.jpg"
    path.write_bytes((await client.get(low["outputs"][0]["download_path"])).content)
    assert low["quality"] == 40 and _identify(path, "%Q") == "40"


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
async def test_quality_is_refused_for_lossless_images(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "plate.png"
    subprocess.run(["convert", "-size", "300x200", "xc:white", str(src)], check=True)
    pid = await make_project("resize-quality-png")
    aid = await _upload(client, pid, src, "image/png")
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "image.resize", "asset_id": aid,
        "input": {"width": 150, "quality": 80},
    })
    assert resp.status_code == 422
    assert "JPEG and WebP" in resp.json()["error"]["message"]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_video_fit_modes(client: AsyncClient, make_project, tmp_path: Path) -> None:
    src = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=duration=1:size=640x360:rate=10", "-pix_fmt", "yuv420p",
                    str(src)], check=True)
    pid = await make_project("resize-fit-video")
    aid = await _upload(client, pid, src, "video/mp4")
    stretch = await _job(client, aid, "video.resize", {"width": 400, "height": 400})
    assert (stretch["width"], stretch["height"]) == (400, 400)
    assert stretch["aspect_ratio_change"] == pytest.approx(-0.4375, abs=1e-3)
    contain = await _job(client, aid, "video.resize",
                         {"width": 400, "height": 400, "fit": "contain"})
    assert (contain["width"], contain["height"]) == (400, 224)
    cover = await _job(client, aid, "video.resize",
                       {"width": 400, "height": 400, "fit": "cover"})
    assert (cover["width"], cover["height"]) == (400, 400)
    assert cover["aspect_ratio_change"] == 0  # cropped, not distorted


async def _rejected(client: AsyncClient, asset_id: str, capability: str, params: dict) -> None:
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": capability, "asset_id": asset_id, "input": params,
    })
    assert resp.status_code == 422, resp.text


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
async def test_image_one_side_with_default_fit_keeps_aspect(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    # The frontend seeds fit "stretch" from the schema default and omits an empty side.
    src = tmp_path / "plate.jpg"
    subprocess.run(["convert", "-size", "2400x1600", "gradient:navy-gold", str(src)], check=True)
    pid = await make_project("resize-one-side-image")
    aid = await _upload(client, pid, src, "image/jpeg")

    by_width = await _job(client, aid, "image.resize", {"width": 1600, "fit": "stretch"})
    assert (by_width["width"], by_width["height"], by_width["fit"]) == (1600, 1067, "stretch")
    assert abs(by_width["aspect_ratio_change"]) < 0.001
    assert by_width["outputs"][0]["filename"] == "plate_resize_1600x1067.jpg"

    by_height = await _job(client, aid, "image.resize", {"height": 1000, "fit": "stretch"})
    assert (by_height["width"], by_height["height"]) == (1500, 1000)
    assert by_height["aspect_ratio_change"] == 0

    for params in ({"fit": "stretch"}, {"width": 1600, "fit": "contain"},
                   {"height": 1000, "fit": "cover"}):
        await _rejected(client, aid, "image.resize", params)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_video_one_side_with_default_fit_keeps_aspect(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=duration=1:size=640x360:rate=10", "-pix_fmt", "yuv420p",
                    str(src)], check=True)
    pid = await make_project("resize-one-side-video")
    aid = await _upload(client, pid, src, "video/mp4")

    by_height = await _job(client, aid, "video.resize", {"height": 180, "fit": "stretch"})
    assert (by_height["width"], by_height["height"], by_height["fit"]) == (320, 180, "stretch")
    assert by_height["aspect_ratio_change"] == 0

    by_width = await _job(client, aid, "video.resize", {"width": 400, "fit": "stretch"})
    # 640x360 at width 400 is 225 high; video sizes are rounded to even numbers.
    assert by_width["width"] == 400 and by_width["height"] in (224, 226)
    assert abs(by_width["aspect_ratio_change"]) < 0.01

    for params in ({"fit": "stretch"}, {"width": 400, "fit": "cover"},
                   {"height": 180, "fit": "contain"}):
        await _rejected(client, aid, "video.resize", params)
