"""video.crop / video.thumbnail source-bounds validation."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.capabilities import registry
from app.capabilities.validation import (
    CapabilityInputValidationError,
    validate_capability_input,
)
from app.core.errors import ValidationError
from app.tools import ffmpeg
from tests.test_video_trim import _make_video, _upload_video

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


# --- schema + pure checks (no binaries) -------------------------------------


@pytest.mark.parametrize(
    ("capability", "params"),
    [
        ("video.crop", {"x": -1, "y": 0, "width": 10, "height": 10}),
        ("video.crop", {"x": 0, "y": -5, "width": 10, "height": 10}),
        ("video.thumbnail", {"timestamp": -0.5}),
    ],
)
def test_negative_values_rejected_by_schema(capability: str, params: dict) -> None:
    with pytest.raises(CapabilityInputValidationError):
        validate_capability_input(registry.get(capability), params)


def test_crop_rectangle_must_fit_frame() -> None:
    check = ffmpeg.check_crop_rectangle
    check(0, 0, 320, 240, frame_width=320, frame_height=240)
    check(100, 40, 220, 200, frame_width=320, frame_height=240)
    with pytest.raises(ValidationError, match=r"fit inside the video \(320×240\)"):
        check(100, 100, 300, 200, frame_width=320, frame_height=240)
    with pytest.raises(ValidationError, match="fit inside"):
        check(0, 1, 320, 240, frame_width=320, frame_height=240)


def test_thumbnail_time_must_be_inside_video() -> None:
    check = ffmpeg.check_thumbnail_time
    check(0, duration=2)
    check(1.99, duration=2)
    check(50, duration=None)  # Unknown duration defers to execution.
    with pytest.raises(ValidationError, match="before the end"):
        check(2, duration=2)
    with pytest.raises(ValidationError, match="negative"):
        check(-1, duration=2)


# --- execution (needs ffmpeg) -----------------------------------------------


def _rotate(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c", "copy",
         "-metadata:s:v", "rotate=90", str(dst)],
        check=True, capture_output=True,
    )


def _size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return stream["width"], stream["height"]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_crop_uses_displayed_size_and_rejects_silent_shift(tmp_path: Path) -> None:
    src, rotated = tmp_path / "in.mp4", tmp_path / "rot.mp4"
    _make_video(src)
    _rotate(src, rotated)
    assert await ffmpeg.probe_display_size(str(src)) == (320, 240)
    assert await ffmpeg.probe_display_size(str(rotated)) == (240, 320)

    dst = tmp_path / "out.mp4"
    with pytest.raises(ValidationError, match="fit inside"):
        await ffmpeg.video_crop(str(src), str(dst), 100, 100, 300, 200)
    with pytest.raises(ValidationError, match=r"\(240×320\)"):
        await ffmpeg.video_crop(str(rotated), str(dst), 0, 0, 320, 240)
    assert not dst.exists()

    await ffmpeg.video_crop(str(rotated), str(dst), 0, 0, 240, 320)
    assert _size(dst) == (240, 320)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_thumbnail_after_last_frame_returns_last_frame(tmp_path: Path) -> None:
    src = tmp_path / "in.mp4"
    _make_video(src)  # 2 s at 15 fps: last frame starts at ~1.933 s
    for name, timestamp in (("mid.jpg", 1.0), ("tail.jpg", 1.99)):
        dst = tmp_path / name
        await ffmpeg.video_thumbnail(str(src), str(dst), timestamp)
        assert dst.stat().st_size > 0
        assert _size(dst) == (320, 240)
    missing = tmp_path / "late.jpg"
    with pytest.raises(ValidationError, match="before the end"):
        await ffmpeg.video_thumbnail(str(src), str(missing), 5)
    assert not missing.exists()


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize(
    ("capability", "params", "message"),
    [
        ("video.crop", {"x": 100, "y": 100, "width": 300, "height": 200}, "fit inside"),
        ("video.crop", {"x": 0, "y": 0, "width": 400, "height": 400}, r"(320×240)"),
        ("video.thumbnail", {"timestamp": 5}, "before the end of the video"),
    ],
)
async def test_admission_rejects_out_of_bounds_requests(
    client: AsyncClient, make_project, tmp_path: Path,
    capability: str, params: dict, message: str,
) -> None:
    src = tmp_path / "in.mp4"
    _make_video(src)
    pid = await make_project("bounds")
    aid = await _upload_video(client, pid, src)
    resp = await client.post(
        f"{API}/jobs", json={"capability_id": capability, "asset_id": aid, "input": params}
    )
    assert resp.status_code == 422, resp.text
    assert message in resp.json()["error"]["message"]
    assert (await client.get(f"{API}/projects/{pid}/jobs")).json()["total"] == 0


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_portrait_crop_of_rotated_video_is_admitted_and_succeeds(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src, rotated = tmp_path / "in.mp4", tmp_path / "rot.mp4"
    _make_video(src)
    _rotate(src, rotated)
    pid = await make_project("rotated-crop")
    aid = await _upload_video(client, pid, rotated)
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "video.crop", "asset_id": aid,
        "input": {"x": 0, "y": 80, "width": 240, "height": 240},
    })
    assert resp.status_code == 201, resp.text
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded", job
    body = (await client.get(job["output"]["outputs"][0]["download_path"])).content
    out = tmp_path / "out.mp4"
    out.write_bytes(body)
    assert _size(out) == (240, 240)
