"""Container-aware normalization, signal ordering and truthful publication."""

from __future__ import annotations

import array
import asyncio
import json
import math
import mimetypes
import shutil
import subprocess
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.errors import ToolError
from app.models.asset import Asset
from app.services.media_metadata import MetadataProbeError, probe_media
from app.services.storage import get_storage
from app.tools import ffmpeg

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs media tools"
)


def run(args: list[str]) -> bytes:
    return subprocess.run(args, check=True, capture_output=True, timeout=30).stdout


def info(path: Path) -> dict:
    return json.loads(run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ]))


@pytest.fixture(scope="module")
def sources(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("concat-sources")
    result = {}
    specs = [
        (f"{label}.{ext}", color, tone, "96x64", 25, None)
        for ext in ("mp4", "webm", "mov", "mkv")
        for label, color, tone in (("a", "red", 440), ("b", "blue", 880),
                                   ("silent", "blue", None))
    ]
    specs += [
        ("heterogeneous.mp4", "blue", 880, "128x96", 15, "libmp3lame"),
        ("heterogeneous.webm", "blue", 880, "128x96", 15, "libopus"),
    ]
    for name, color, tone, size, fps, audio_encoder in specs:
        path = root / name
        webm = path.suffix == ".webm"
        args = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                f"color=c={color}:s={size}:r={fps}:d=1"]
        if tone is not None:
            args += ["-f", "lavfi", "-i",
                     f"sine=frequency={tone}:sample_rate=48000:duration=1"]
        args += ["-c:v", "libvpx-vp9" if webm else "libx264", "-pix_fmt", "yuv420p"]
        if tone is not None:
            args += ["-c:a", audio_encoder or ("libopus" if webm else "aac")]
        run([*args, str(path)])
        result[name] = path
    return result


def assert_signal(path: Path, start: float, tone: int | None) -> None:
    samples = array.array("f", run([
        "ffmpeg", "-v", "error", "-i", str(path), "-ss", str(start), "-t", "0.4",
        "-map", "0:a:0", "-ac", "1", "-ar", "8000", "-f", "f32le", "pipe:1",
    ]))
    assert len(samples) >= 3000
    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    if tone is None:
        assert rms < 0.001
    else:
        assert rms > 0.03
        crossings = sum(a <= 0 < b for a, b in zip(samples, samples[1:], strict=False))
        assert abs(crossings * 8000 / len(samples) - tone) < 15


def assert_color(path: Path, start: float, channel: int) -> None:
    pixel = run([
        "ffmpeg", "-v", "error", "-i", str(path), "-ss", str(start), "-frames:v", "1",
        "-vf", "scale=1:1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ])
    assert len(pixel) == 3
    assert pixel[channel] > max(pixel[i] for i in range(3) if i != channel) + 100


CASES = [
    ("a.mp4", "b.mp4", 440, 880), ("a.webm", "b.webm", 440, 880),
    ("a.mp4", "b.webm", 440, 880), ("a.webm", "b.mp4", 440, 880),
    ("silent.mp4", "silent.mp4", None, None),
    ("silent.webm", "silent.webm", None, None),
    ("a.mp4", "silent.webm", 440, None), ("silent.mp4", "b.webm", None, 880),
    ("a.webm", "silent.mp4", 440, None), ("silent.webm", "b.mp4", None, 880),
    ("a.mp4", "heterogeneous.webm", 440, 880),
    ("a.webm", "heterogeneous.mp4", 440, 880),
    ("a.mov", "b.webm", 440, 880), ("a.mkv", "b.webm", 440, 880),
]


@pytest.mark.parametrize("first,second,tone_a,tone_b", CASES)
async def test_concat_job(sources, client, make_project, monkeypatch,
                          first, second, tone_a, tone_b) -> None:
    pid = await make_project()
    ids = []
    for name in (first, second):
        response = await client.post(f"/api/v1/projects/{pid}/assets", files={
            "file": (name, sources[name].read_bytes(), mimetypes.guess_type(name)[0]),
        })
        assert response.status_code == 201, response.text
        ids.append(response.json()["id"])
    commands = []
    actual_run = ffmpeg.runner.run

    async def capture(args, **kwargs):
        commands.append(args)
        return await actual_run(args, **kwargs)

    monkeypatch.setattr(ffmpeg.runner, "run", capture)
    response = await client.post("/api/v1/jobs", json={
        "capability_id": "video.concat", "project_id": pid, "input": {"asset_ids": ids},
    })
    assert response.status_code == 201, response.text
    job = (await client.get(f"/api/v1/jobs/{response.json()['id']}")).json()
    assert job["status"] == "succeeded", job.get("error")
    assert job["output"]["operation"] == "video.concat"
    assert job["output"]["inputs"] == 2
    assert len(job["output"]["outputs"]) == 1
    output = job["output"]["outputs"][0]
    assert (await client.get(output["download_path"])).status_code == 200
    asset = (await client.get(f"/api/v1/assets/{output['asset_id']}")).json()
    container = sources[first].suffix.lstrip(".")
    webm = container == "webm"
    assert asset["extension"] == container
    assert asset["original_filename"].endswith(f"_concat.{container}")
    assert output["filename"] == asset["original_filename"]
    assert asset["mime_type"] == {
        "mp4": "video/mp4", "webm": "video/webm", "mov": "video/quicktime",
        # Preserve the existing platform-dependent MIME fallback for MKV.
        "mkv": mimetypes.guess_type("output.mkv")[0] or "application/octet-stream",
    }[container]
    assert asset["technical_metadata_status"] == "available"
    assert asset["video_codec"] == ("vp9" if webm else "h264")
    assert asset["audio_codec"] == ("opus" if webm else "aac")
    assert asset["audio_sample_rate"] == (48000 if webm else 44100)
    assert asset["audio_channels"] == 2
    assert (asset["width"], asset["height"]) == (96, 64)
    assert 1.9 < asset["duration_seconds"] < 2.15  # Includes encoder/muxer padding.
    path = get_storage().get_path(asset["storage_path"])
    metadata = info(path)
    assert ("matroska" if container == "mkv" else container) in metadata["format"][
        "format_name"].split(",")
    assert asset["container_format"] == metadata["format"]["format_name"]
    run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    assert_signal(path, 0.25, tone_a)
    assert_signal(path, 1.25, tone_b)
    assert_color(path, 0.25, 2 if first.startswith("silent") else 0)
    assert_color(path, 1.25, 2)
    normalizations = [cmd for cmd in commands if "_concat_norm_" in cmd[-1]]
    assert len(normalizations) == 2
    assert all(cmd[-1].endswith(f".{container}") for cmd in normalizations)
    assert all(cmd[cmd.index("-vf") + 1].endswith("fps=30") for cmd in normalizations)
    concat = next(cmd for cmd in commands if "concat" in cmd)
    assert concat[concat.index("-c") + 1] == "copy"
    assert all(not Path(cmd[-1]).exists() for cmd in normalizations)
    async with async_session_factory() as session:
        derived = list(await session.scalars(select(Asset).where(
            Asset.project_id == uuid.UUID(pid), Asset.source_asset_id.is_not(None))))
        assert [str(a.id) for a in derived] == [output["asset_id"]]


@pytest.mark.parametrize("container", ["mp4", "webm"])
async def test_success_cleans_only_intermediates(sources, tmp_path, container) -> None:
    output = tmp_path / f"out.{container}"
    await ffmpeg.video_concat(
        [str(sources[f"a.{container}"]), str(sources[f"b.{container}"])],
        str(output), tmp_path,
    )
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("stage", ["normalization", "mux", "validation", "cancellation"])
async def test_failure_cleanup(sources, tmp_path, monkeypatch, stage) -> None:
    actual_run = ffmpeg.runner.run

    async def fail(args, **kwargs):
        if ((stage in {"normalization", "cancellation"} and "_concat_norm_001" in args[-1])
                or (stage == "mux" and "concat" in args)):
            Path(args[-1]).write_bytes(b"partial output")
            if stage == "cancellation":
                raise asyncio.CancelledError
            raise ToolError("Controlled concat failure")
        return await actual_run(args, **kwargs)

    monkeypatch.setattr(ffmpeg.runner, "run", fail)
    if stage == "validation":
        async def invalid(*args, **kwargs):
            raise MetadataProbeError("tool_failed")
        monkeypatch.setattr(ffmpeg, "probe_media", invalid)
    expected = asyncio.CancelledError if stage == "cancellation" else ToolError
    with pytest.raises(expected):
        await ffmpeg.video_concat([str(sources["a.webm"]), str(sources["b.mp4"])],
                                  str(tmp_path / "out.webm"), tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fault", ["missing", "empty", "container", "video_codec",
                                   "audio_codec", "no_video", "no_audio", "duration"])
async def test_semantic_validation(sources, tmp_path, monkeypatch, fault) -> None:
    actual_probe = probe_media

    async def invalid(path, media_type):
        metadata = await actual_probe(path, media_type)
        changes = {
            "container": {"container_format": "avi"}, "video_codec": {"video_codec": "mpeg4"},
            "audio_codec": {"audio_codec": "mp3"}, "no_video": {"has_video_stream": False},
            "no_audio": {"has_audio_stream": False}, "duration": {"duration_seconds": 0},
        }
        return replace(metadata, **changes[fault])

    if fault in {"missing", "empty"}:
        actual_run = ffmpeg.runner.run
        async def broken(args, **kwargs):
            result = await actual_run(args, **kwargs)
            if "concat" in args:
                if fault == "missing":
                    Path(args[-1]).unlink()
                else:
                    Path(args[-1]).write_bytes(b"")
            return result
        monkeypatch.setattr(ffmpeg.runner, "run", broken)
    else:
        monkeypatch.setattr(ffmpeg, "probe_media", invalid)
    with pytest.raises(ToolError, match="Concat output"):
        await ffmpeg.video_concat([str(sources["a.mp4"]), str(sources["b.mp4"])],
                                  str(tmp_path / "out.mp4"), tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", ["normalization", "mux", "validation"])
async def test_failed_job_publishes_nothing(sources, client, make_project, monkeypatch,
                                          stage) -> None:
    pid = await make_project()
    ids = []
    for name in ("a.webm", "b.mp4"):
        response = await client.post(f"/api/v1/projects/{pid}/assets",
                                    files={"file": (name, sources[name].read_bytes())})
        assert response.status_code == 201
        ids.append(response.json()["id"])
    actual = ffmpeg.runner.run
    async def fail(args, **kwargs):
        if ((stage == "normalization" and "_concat_norm_001" in args[-1])
                or (stage == "mux" and "concat" in args)):
            raise ToolError("Controlled concat failure")
        return await actual(args, **kwargs)
    monkeypatch.setattr(ffmpeg.runner, "run", fail)
    if stage == "validation":
        async def invalid(*args, **kwargs):
            raise MetadataProbeError("tool_failed")
        monkeypatch.setattr(ffmpeg, "probe_media", invalid)
    response = await client.post("/api/v1/jobs", json={
        "capability_id": "video.concat", "project_id": pid, "input": {"asset_ids": ids},
    })
    assert response.status_code == 201
    job = (await client.get(f"/api/v1/jobs/{response.json()['id']}")).json()
    assert job["status"] == "failed"
    assert job["output"] is None
    async with async_session_factory() as session:
        assets = list(await session.scalars(
            select(Asset).where(Asset.project_id == uuid.UUID(pid))))
        assert {str(a.id) for a in assets} == set(ids)
    project_dir = Path(get_storage().get_path(f"projects/{pid}/assets"))
    assert sorted(p.name for p in project_dir.iterdir()) == sorted(ids)
