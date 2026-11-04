"""video.export: single-asset delivery presets (D-08) with source bitrate caps."""

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
from app.composition.export_presets import resolve_effective_settings
from app.tools import ffmpeg
from tests.test_video_trim import _upload_video

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def _settings(preset: str, social_format: str | None = None):
    request = {"preset": preset}
    if social_format:
        request["social_format"] = social_format
    return resolve_effective_settings(request, canvas_width=0, canvas_height=0)


# --- schema (no binaries) ---------------------------------------------------


def test_schema_offers_named_destinations_only() -> None:
    cap = registry.get("video.export")
    assert cap.input_schema["properties"]["preset"]["enum"] == [
        "web", "social", "preview", "archival",
    ]
    for params in ({"preset": "web"}, {"preset": "archival"},
                   {"preset": "social", "social_format": "portrait"}):
        validate_capability_input(cap, params)
    for params in ({}, {"preset": "advanced"}, {"preset": "social"},
                   {"preset": "web", "social_format": "square"}):
        with pytest.raises(CapabilityInputValidationError):
            validate_capability_input(cap, params)
    assert cap.summary and all(
        "x-help" in prop for prop in cap.input_schema["properties"].values()
    )


# --- media helpers -----------------------------------------------------------


def _encode(path: Path, *, size: str, seconds: int = 2, crf: int = 23,
            audio: bool = True, extra: list[str] | None = None) -> None:
    args = ["ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=duration={seconds}:size={size}:rate=25"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    args += extra or ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-b:a", "64k", "-shortest"]
    subprocess.run([*args, str(path)], check=True, capture_output=True)


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height:format=format_name",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def _video(info: dict) -> dict:
    return next(s for s in info["streams"] if s["codec_type"] == "video")


# --- execution (needs ffmpeg) -----------------------------------------------


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_web_preset_is_uncapped_and_inflation_is_reported(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    # Product decision: presets are quality-targeted with no size cap. An
    # efficiently encoded source can grow; size_comparison makes that visible.
    src = tmp_path / "efficient.mp4"
    _encode(src, size="1280x720", seconds=4, crf=40)
    plain, exported = tmp_path / "plain.mp4", tmp_path / "exported.mp4"
    await ffmpeg.export_delivery_transform(str(src), str(plain), _settings("web"))
    await ffmpeg.video_export(str(src), str(exported), _settings("web"))
    assert exported.stat().st_size == plain.stat().st_size

    pid = await make_project("video-export-inflation")
    aid = await _upload_video(client, pid, src)
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "video.export", "asset_id": aid, "input": {"preset": "web"},
    })
    assert resp.status_code == 201, resp.text
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded", job
    comparison = job["output"]["size_comparison"]
    assert comparison["delta_bytes"] > 0 and comparison["ratio"] > 1


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize(
    ("preset", "social_format", "source", "expected_size", "codecs", "container"),
    [
        ("preview", None, "1920x1080", (1280, 720), {"h264", "aac"}, "mov,mp4"),
        ("web", None, "640x360", (640, 360), {"h264", "aac"}, "mov,mp4"),  # never upscaled
        ("social", "portrait", "640x360", (1080, 1920), {"h264", "aac"}, "mov,mp4"),
        ("social", "square", "640x360", (1080, 1080), {"h264", "aac"}, "mov,mp4"),
        ("archival", None, "320x240", (320, 240), {"ffv1", "flac"}, "matroska"),
    ],
)
async def test_presets_produce_their_delivery_format(
    tmp_path: Path, preset: str, social_format: str | None, source: str,
    expected_size: tuple[int, int], codecs: set[str], container: str,
) -> None:
    src = tmp_path / "in.mp4"
    _encode(src, size=source)
    dst = tmp_path / f"out.{_settings(preset, social_format).container.value}"
    await ffmpeg.video_export(str(src), str(dst), _settings(preset, social_format))
    info = _probe(dst)
    assert info["format"]["format_name"].startswith(container)
    assert {s["codec_name"] for s in info["streams"]} == codecs
    assert (_video(info)["width"], _video(info)["height"]) == expected_size


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_silent_video_exports(tmp_path: Path) -> None:
    src, dst = tmp_path / "silent.mp4", tmp_path / "out.mp4"
    _encode(src, size="320x240", audio=False)
    await ffmpeg.video_export(str(src), str(dst), _settings("web"))
    assert [s["codec_type"] for s in _probe(dst)["streams"]] == ["video"]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_export_job_publishes_named_copy_with_size_comparison(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "clip.mp4"
    _encode(src, size="640x360")
    pid = await make_project("video-export")
    aid = await _upload_video(client, pid, src)
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "video.export", "asset_id": aid,
        "input": {"preset": "social", "social_format": "square"},
    })
    assert resp.status_code == 201, resp.text
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded", job
    assert job["output"]["preset"] == "social"
    assert job["output"]["outputs"][0]["filename"] == "clip_social-square.mp4"
    assert "size_comparison" in job["output"]


@pytest.mark.parametrize(
    "params",
    [{"preset": "social"}, {"preset": "archival", "social_format": "square"},
     {"preset": "advanced"}],
)
async def test_invalid_preset_requests_are_rejected_before_job_creation(
    client: AsyncClient, make_project, params: dict
) -> None:
    pid = await make_project("video-export-422")
    aid = (await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": ("clip.mp4", b"not media", "video/mp4")},
    )).json()["id"]
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "video.export", "asset_id": aid, "input": params,
    })
    assert resp.status_code == 422, resp.text
    assert (await client.get(f"{API}/projects/{pid}/jobs")).json()["total"] == 0
