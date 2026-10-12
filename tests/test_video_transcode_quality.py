"""Public quality admission, legacy argv and real codec/container regressions."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from jsonschema import Draft202012Validator
from sqlalchemy import func, select

from app.agent.catalog import build_catalog, catalog_text
from app.capabilities import registry
from app.capabilities.media_options import VIDEO_TRANSCODE_OPTIONS
from app.capabilities.validation import validate_capability_input
from app.core.database import async_session_factory
from app.core.errors import ToolError
from app.models.asset import Asset
from app.models.job import Job
from app.tools import ffmpeg, runner

API = "/api/v1"
LEVELS = ("high", "balanced", "compressed")
EXPECTED_CRF = {
    "libx264": (18, 22, 28),
    "libx265": (22, 28, 34),
    "libvpx-vp9": (24, 32, 40),
}
CODEC_NAMES = {
    "libx264": "h264", "libx265": "hevc", "libvpx-vp9": "vp9",
    "aac": "aac", "libmp3lame": "mp3", "libopus": "opus",
    "libvorbis": "vorbis", "flac": "flac",
}
NEEDS_TOOLS = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="requires real FFmpeg and ffprobe",
)


def _run(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _make_source(path: Path, *, duration: float = 1.2, size: str = "160x96") -> None:
    _run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
        f"testsrc2=size={size}:rate=20:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264", "-crf", "12", "-c:a", "aac", "-shortest", str(path),
    ])


def _probe(path: Path) -> dict:
    return json.loads(_run([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path),
    ]))


@pytest.fixture(scope="module")
def source_video(tmp_path_factory) -> Path:
    if not shutil.which("ffmpeg"):
        pytest.skip("requires FFmpeg")
    path = tmp_path_factory.mktemp("transcode-quality") / "source.mp4"
    _make_source(path)
    return path


async def test_quality_public_schema_and_agent_share_optional_control(client: AsyncClient) -> None:
    detail = await client.get(f"{API}/capabilities/video.transcode")
    assert detail.status_code == 200
    schema = detail.json()["input_schema"]
    quality = schema["properties"]["quality"]
    assert quality["type"] == "string"
    assert quality["title"] == "Video quality"
    assert quality["enum"] == list(LEVELS)
    assert set(quality["x-enum-descriptions"]) == set(LEVELS)
    assert "video_codec" in quality["description"]
    assert "Omit" in quality["description"]
    assert "size" in quality["description"]
    assert "default" not in quality
    assert schema["required"] == ["format"]
    assert not {"crf", "target_size", "target_size_mb", "target_filesize", "max_file_size"} & (
        schema["properties"].keys()
    )
    listed = (await client.get(f"{API}/capabilities")).json()
    assert next(c for c in listed if c["id"] == "video.transcode")["input_schema"] == schema
    catalog = next(c for c in build_catalog() if c["capability_id"] == "video.transcode")
    assert catalog["input_schema"] == schema
    assert "quality: string optional" in catalog_text()
    for container, choices in VIDEO_TRANSCODE_OPTIONS.items():
        validate_capability_input(registry.get("video.transcode"), {"format": container})
        for codec in choices["video_codec"]:
            for level in LEVELS:
                validate_capability_input(registry.get("video.transcode"), {
                    "format": container, "video_codec": codec, "quality": level,
                })


async def test_video_quality_retry_identity_separates_prequality_cache_entries(
    client: AsyncClient, make_project, monkeypatch,
) -> None:
    from app.agent.reuse import REUSE_VERSIONS, build_execution_identity

    project_id = uuid.UUID(await make_project())
    upload = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("source.mp4", b"cache-identity-input", "video/mp4")},
    )
    assert upload.status_code == 201, upload.text
    asset_id = uuid.UUID(upload.json()["id"])

    async with async_session_factory() as session:
        asset = await session.get(Asset, asset_id)
        assert asset is not None
        asset.checksum_sha256 = "a" * 64
        await session.commit()

        legacy_parameters = {
            "format": "mp4", "video_codec": "libx264", "audio_codec": "aac",
        }
        legacy_identity = await build_execution_identity(
            session,
            project_id=project_id,
            capability_id="video.transcode",
            asset_id=asset_id,
            params=legacy_parameters,
        )
        omitted_again = await build_execution_identity(
            session,
            project_id=project_id,
            capability_id="video.transcode",
            asset_id=asset_id,
            params=dict(legacy_parameters),
        )
        high_identity = await build_execution_identity(
            session,
            project_id=project_id,
            capability_id="video.transcode",
            asset_id=asset_id,
            params={**legacy_parameters, "quality": "high"},
        )
        compressed_identity = await build_execution_identity(
            session,
            project_id=project_id,
            capability_id="video.transcode",
            asset_id=asset_id,
            params={**legacy_parameters, "quality": "compressed"},
        )

    assert REUSE_VERSIONS["video.transcode"] == "2"
    assert legacy_identity == omitted_again
    assert legacy_identity != high_identity
    assert high_identity != compressed_identity

    # Before quality existed, JSON Schema allowed extra properties by default.
    # An Agent request could therefore persist quality=high in an old Job while
    # the old handler ignored it. That old cached result had the same parameters
    # and version as a new quality-aware request unless the reuse version changes.
    old_schema = deepcopy(registry.get("video.transcode").input_schema)
    del old_schema["properties"]["quality"]
    old_schema["allOf"].pop()
    Draft202012Validator(old_schema).validate({**legacy_parameters, "quality": "high"})

    from app.agent import reuse

    async with async_session_factory() as session:
        monkeypatch.setitem(reuse.REUSE_VERSIONS, "video.transcode", "1")
        old_ignored_quality_identity = await build_execution_identity(
            session,
            project_id=project_id,
            capability_id="video.transcode",
            asset_id=asset_id,
            params={**legacy_parameters, "quality": "high"},
        )
        monkeypatch.setitem(reuse.REUSE_VERSIONS, "video.transcode", "2")
        new_quality_identity = await build_execution_identity(
            session,
            project_id=project_id,
            capability_id="video.transcode",
            asset_id=asset_id,
            params={**legacy_parameters, "quality": "high"},
        )
    assert old_ignored_quality_identity != new_quality_identity


@pytest.mark.parametrize("params", [
    {"format": "mp4", "video_codec": "libx264", "quality": "ultra"},
    {"format": "mp4", "video_codec": "libx264", "quality": 22},
    {"format": "mp4", "video_codec": "libx264", "quality": None},
    {"format": "mp4", "video_codec": "libx264", "quality": True},
    {"format": "mp4", "video_codec": "libx264", "quality": ""},
    {"format": "mp4", "quality": "balanced"},
    {"format": "webm", "quality": "compressed"},
    {"format": "mp4", "video_codec": "libx265", "quality": "high"},
    {"format": "webm", "video_codec": "libx264", "quality": "balanced"},
    {"format": "webm", "video_codec": "libvpx-vp9", "audio_codec": "aac", "quality": "high"},
])
async def test_invalid_quality_rejected_before_job_creation(client, make_project, params) -> None:
    project_id = await make_project()
    upload = await client.post(f"{API}/projects/{project_id}/assets", files={
        "file": ("source.mp4", b"validation-only", "video/mp4"),
    })
    assert upload.status_code == 201
    response = await client.post(f"{API}/jobs", json={
        "capability_id": "video.transcode", "asset_id": upload.json()["id"], "input": params,
    })
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "validation_error"
    async with async_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Job)) == 0
    assets = (await client.get(f"{API}/projects/{project_id}/assets")).json()
    assert len(assets) == 1


@pytest.mark.parametrize("video_codec,audio_codec", [
    (None, None), ("libx264", "aac"), ("libx265", "flac"), ("libvpx-vp9", "libopus"),
])
async def test_omitted_quality_retains_exact_legacy_command(monkeypatch, video_codec, audio_codec):
    run = AsyncMock()
    monkeypatch.setattr(runner, "run", run)
    await ffmpeg.video_transcode("/input.mp4", "/output.mkv", video_codec, audio_codec)
    expected = ["ffmpeg", "-y", "-loglevel", "error", "-i", "/input.mp4"]
    if video_codec:
        expected += ["-c:v", video_codec]
    if audio_codec:
        expected += ["-c:a", audio_codec]
    run.assert_awaited_once_with([*expected, "/output.mkv"])


@pytest.mark.parametrize("codec", EXPECTED_CRF)
@pytest.mark.parametrize("quality", LEVELS)
async def test_quality_reaches_encoder_without_audio_or_geometry_changes(
    monkeypatch, codec, quality,
):
    run = AsyncMock()
    monkeypatch.setattr(runner, "run", run)
    await ffmpeg.video_transcode("/input.mp4", "/output.mkv", codec, "aac", quality=quality)
    settings = ["-crf", str(EXPECTED_CRF[codec][LEVELS.index(quality)])]
    if codec == "libvpx-vp9":
        settings += ["-b:v", "0"]
    run.assert_awaited_once_with([
        "ffmpeg", "-y", "-loglevel", "error", "-i", "/input.mp4",
        "-c:v", codec, "-c:a", "aac", *settings, "/output.mkv",
    ])


@pytest.mark.parametrize("codec,quality", [(None, "high"), ("copy", "high"), ("libx264", "ultra")])
async def test_internal_invalid_quality_never_launches_tool(monkeypatch, codec, quality):
    run = AsyncMock()
    monkeypatch.setattr(runner, "run", run)
    with pytest.raises(ToolError):
        await ffmpeg.video_transcode("/input.mp4", "/output.mkv", codec, None, quality=quality)
    run.assert_not_awaited()


@NEEDS_TOOLS
@pytest.mark.parametrize("container,video_codec,audio_codec", [
    (container, video, audio)
    for container, choices in VIDEO_TRANSCODE_OPTIONS.items()
    for video in choices["video_codec"]
    for audio in choices["audio_codec"]
])
@pytest.mark.parametrize("quality", [None, *LEVELS])
async def test_real_quality_codec_container_matrix(
    client, make_project, source_video, tmp_path, container, video_codec, audio_codec, quality,
) -> None:
    project_id = await make_project()
    upload = await client.post(f"{API}/projects/{project_id}/assets", files={
        "file": ("source.mp4", source_video.read_bytes(), "video/mp4"),
    })
    assert upload.status_code == 201, upload.text
    params = {"format": container, "video_codec": video_codec, "audio_codec": audio_codec}
    if quality is not None:
        params["quality"] = quality
    response = await client.post(f"{API}/jobs", json={
        "capability_id": "video.transcode", "asset_id": upload.json()["id"], "input": params,
    })
    assert response.status_code == 201, response.text
    job = (await client.get(f"{API}/jobs/{response.json()['id']}")).json()
    assert job["status"] == "succeeded", job.get("error")
    assert job["input"] == params
    assert len(job["output"]["outputs"]) == 1
    output = job["output"]["outputs"][0]
    download = await client.get(output["download_path"])
    assert download.status_code == 200
    path = tmp_path / f"output.{container}"
    path.write_bytes(download.content)
    probe = _probe(path)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert video["codec_name"] == CODEC_NAMES[video_codec]
    assert audio["codec_name"] == CODEC_NAMES[audio_codec]
    assert (video["width"], video["height"], video["r_frame_rate"]) == (160, 96, "20/1")
    assert float(probe["format"]["duration"]) == pytest.approx(1.2, abs=0.15)
    expected_container = "matroska" if container in {"mkv", "webm"} else container
    assert expected_container in probe["format"]["format_name"].split(",")
    asset = (await client.get(f"{API}/assets/{output['asset_id']}")).json()
    assert asset["technical_metadata_status"] == "available"
    assert asset["duration_seconds"] > 0
    assert asset["provenance_kind"] == "derived"


@NEEDS_TOOLS
@pytest.mark.parametrize("container", VIDEO_TRANSCODE_OPTIONS)
async def test_real_omitted_codecs_and_quality_keep_container_defaults(
    source_video, tmp_path, container,
):
    output = tmp_path / f"default.{container}"
    await ffmpeg.video_transcode(str(source_video), str(output), None, None)
    probe = _probe(output)
    assert float(probe["format"]["duration"]) > 0
    assert {s["codec_type"] for s in probe["streams"]} == {"video", "audio"}


@NEEDS_TOOLS
@pytest.mark.parametrize("codec", EXPECTED_CRF)
async def test_compression_changes_real_encoded_size(tmp_path, codec):
    source = tmp_path / "motion.mp4"
    _make_source(source, duration=3, size="320x180")
    sizes = {}
    for quality in ("high", "compressed"):
        output = tmp_path / f"{quality}.mkv"
        await ffmpeg.video_transcode(str(source), str(output), codec, "aac", quality=quality)
        sizes[quality] = output.stat().st_size
    # Only a controlled-input ordering/margin, never an exact filesize promise.
    assert sizes["compressed"] < sizes["high"] * 0.85, sizes
    print(f"{codec}: high={sizes['high']} compressed={sizes['compressed']} bytes")
