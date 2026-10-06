"""Real-media timestamp splitting and pre-publication semantic validation."""

from __future__ import annotations

import array
import asyncio
import json
import math
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app.agent.catalog import build_catalog
from app.capabilities import registry
from app.capabilities.validation import CapabilityInputValidationError, validate_capability_input
from app.core.database import async_session_factory
from app.core.errors import ToolError, ValidationError
from app.models.asset import Asset
from app.services.media_metadata import MetadataProbeError, probe_video_timeline
from app.services.storage import get_storage
from app.tools import ffmpeg

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
media = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe required")


def run(args: list[str]) -> bytes:
    return subprocess.run(args, check=True, capture_output=True, timeout=30).stdout


def info(path: Path) -> dict:
    return json.loads(run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ]))


def frames(path: Path) -> list[dict]:
    return json.loads(run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time,pkt_duration_time,key_frame",
        "-of", "json", str(path),
    ]))["frames"]


def frame_pixels(path: Path, frame: int) -> bytes:
    return run([
        "ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame})",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])


def audio_samples(path: Path) -> array.array:
    pcm = array.array("f")
    pcm.frombytes(run([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-t", "0.5",
        "-ac", "1", "-ar", "8000", "-f", "f32le", "pipe:1",
    ]))
    return pcm


@pytest.fixture(scope="module")
def sources(tmp_path_factory) -> dict[str, Path]:
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg/ffprobe required")
    root = tmp_path_factory.mktemp("split-sources")
    result = {}
    for name, fps, gop, sound in [
        ("long.mp4", "25", 250, 1), ("friendly.mp4", "25", 25, 1),
        ("long.webm", "25", 250, 1), ("silent.mp4", "25", 250, 0),
        ("fractional.mp4", "30000/1001", 300, 1), ("multi.mp4", "25", 250, 2),
        ("audio-longer.mp4", "25", 250, 1), ("offset.mp4", "25", 250, 1),
    ]:
        path = root / name
        args = [
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
            f"testsrc2=s=96x64:r={fps}:d=4",
        ]
        for index in range(sound):
            audio_duration = 8 if name == "audio-longer.mp4" else 4
            args += [
                "-f", "lavfi", "-i",
                f"sine=frequency={440 * (index + 1)}:sample_rate=48000:duration={audio_duration}",
            ]
        args += ["-map", "0:v:0"]
        for index in range(sound):
            args += ["-map", f"{index + 1}:a:0"]
        if path.suffix == ".webm":
            args += ["-c:v", "libvpx-vp9", "-g", str(gop), "-c:a", "libopus"]
        else:
            args += [
                "-c:v", "libx264", "-g", str(gop), "-keyint_min", str(gop),
                "-sc_threshold", "0", "-bf", "0", "-c:a", "aac",
            ]
        if name == "offset.mp4":
            args += ["-output_ts_offset", "1"]
        run([*args, str(path)])
        result[name] = path
    return result


@media
@pytest.mark.parametrize("name,markers", [
    ("long.mp4", [1]), ("friendly.mp4", [1.3]), ("long.webm", [1.3]),
    ("silent.mp4", [1.3]), ("fractional.mp4", [1.3]), ("multi.mp4", [1.3]),
    ("long.mp4", [1, 2, 3]), ("long.webm", [0.7, 1.3, 2.7]),
    ("long.mp4", [0.001]), ("long.mp4", [3.96]),
    ("audio-longer.mp4", [1.3]), ("offset.mp4", [1.3]),
])
async def test_timestamp_split(sources, tmp_path, name, markers) -> None:
    source = sources[name]
    original = frames(source)
    if name.startswith("long"):
        assert sum(frame["key_frame"] for frame in original) == 1
    origin = float(original[0]["best_effort_timestamp_time"])
    times = [float(frame["best_effort_timestamp_time"]) - origin for frame in original]
    frame_duration = float(original[-1]["pkt_duration_time"])
    duration = times[-1] + frame_duration
    tolerance = max(b - a for a, b in zip(times, times[1:], strict=False)) + 0.002
    edges = [0] + [next(i for i, t in enumerate(times) if t + 1e-7 >= m) for m in markers]
    edges += [len(original)]
    outputs = await ffmpeg.video_split(
        str(source), str(tmp_path / f"seg%03d{source.suffix}"), markers
    )
    assert len(outputs) == len(markers) + 1
    measured_total = 0.0
    for index, output in enumerate(outputs):
        actual = frames(output)
        assert len(actual) == edges[index + 1] - edges[index]
        actual_duration = (
            float(actual[-1]["best_effort_timestamp_time"])
            - float(actual[0]["best_effort_timestamp_time"])
            + float(actual[-1]["pkt_duration_time"])
        )
        expected_end = times[edges[index + 1]] if edges[index + 1] < len(times) else duration
        assert actual_duration == pytest.approx(expected_end - times[edges[index]], abs=tolerance)
        measured_total += actual_duration
        # First-frame identity proves chronological ownership, not merely lengths.
        expected_pixels = frame_pixels(source, edges[index])
        actual_pixels = frame_pixels(output, 0)
        assert len(actual_pixels) == len(expected_pixels) > 0
        assert sum(abs(a - b) for a, b in zip(
            actual_pixels, expected_pixels, strict=True
        )) / len(
            actual_pixels
        ) < 10
        streams = info(output)["streams"]
        video = next(stream for stream in streams if stream["codec_type"] == "video")
        assert video["codec_name"] == ("vp9" if source.suffix == ".webm" else "h264")
        audio = [stream for stream in streams if stream["codec_type"] == "audio"]
        if name == "silent.mp4":
            assert not audio
        else:
            assert len(audio) == 1
            assert audio[0]["codec_name"] == ("opus" if source.suffix == ".webm" else "aac")
            pcm = audio_samples(output)
            rms = math.sqrt(sum(value * value for value in pcm) / len(pcm))
            assert rms > 0.01
            if name == "multi.mp4":
                crossings = sum(a <= 0 < b for a, b in zip(pcm, pcm[1:], strict=False))
                assert crossings * 8000 / len(pcm) == pytest.approx(440, abs=20)
            if "duration" in audio[0]:
                assert float(audio[0]["duration"]) == pytest.approx(actual_duration, abs=0.08)
    assert measured_total == pytest.approx(duration, abs=len(outputs) * tolerance)
    for marker, edge in zip(markers, edges[1:-1], strict=True):
        assert 0 <= times[edge] - marker <= tolerance


@pytest.mark.parametrize("markers", [
    [], [0], [-1], [1, 1], [2, 1], [float("nan")], [float("inf")],
    [-float("inf")], [True], ["1"], [10**1000],
])
async def test_invalid_static_markers_do_not_probe_or_render(monkeypatch, tmp_path, markers):
    async def unexpected(*args, **kwargs):
        raise AssertionError("Invalid markers must fail before probing/rendering")
    monkeypatch.setattr(ffmpeg, "probe_media", unexpected)
    monkeypatch.setattr(ffmpeg.runner, "run", unexpected)
    with pytest.raises(ValidationError):
        await ffmpeg.video_split("unused.mp4", str(tmp_path / "seg%03d.mp4"), markers)
    assert not list(tmp_path.iterdir())


@media
@pytest.mark.parametrize("markers", [[4], [5], [3.999], [1.001, 1.002]])
async def test_duration_and_empty_intervals_rejected_before_render(sources, tmp_path, monkeypatch,
                                                                 markers):
    async def unexpected(*args, **kwargs):
        raise AssertionError("Invalid boundaries must fail before rendering")
    monkeypatch.setattr(ffmpeg.runner, "run", unexpected)
    with pytest.raises(ValidationError, match="before the end|non-empty video intervals"):
        await ffmpeg.video_split(str(sources["long.mp4"]), str(tmp_path / "seg%03d.mp4"), markers)
    assert not list(tmp_path.iterdir())


@media
@pytest.mark.parametrize("failure", [
    "render", "cancel", "wrong_output", "probe", "missing", "empty", "corrupt",
])
async def test_partial_failure_removes_all_split_files(sources, tmp_path, monkeypatch, failure):
    real_run = ffmpeg.runner.run
    real_probe = ffmpeg.probe_video_timeline
    calls = 0

    async def render(args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2 and failure in ("render", "cancel"):
            Path(args[-1]).write_bytes(b"partial")
            if failure == "cancel":
                raise asyncio.CancelledError
            raise ToolError("Simulated rendering failure")
        if calls == 2 and failure == "wrong_output":
            shutil.copyfile(sources["long.mp4"], args[-1])
        elif calls == 2 and failure in ("missing", "empty", "corrupt"):
            if failure != "missing":
                Path(args[-1]).write_bytes(b"bad media" if failure == "corrupt" else b"")
        else:
            await real_run(args, **kwargs)

    async def probe(path, *args, **kwargs):
        if path.name == "seg001.mp4" and failure == "probe":
            raise MetadataProbeError("invalid_probe_output")
        return await real_probe(path, *args, **kwargs)

    monkeypatch.setattr(ffmpeg.runner, "run", render)
    monkeypatch.setattr(ffmpeg, "probe_video_timeline", probe)
    with pytest.raises((ToolError, asyncio.CancelledError)):
        await ffmpeg.video_split(str(sources["long.mp4"]), str(tmp_path / "seg%03d.mp4"), [1])
    assert not list(tmp_path.iterdir())


@media
async def test_corrupt_source_fails_cleanly(tmp_path):
    source = tmp_path / "corrupt.mp4"
    source.write_bytes(b"not a movie")
    with pytest.raises(ValidationError, match="reliable source video timing"):
        await ffmpeg.video_split(str(source), str(tmp_path / "seg%03d.mp4"), [1])
    assert list(tmp_path.iterdir()) == [source]


@media
@pytest.mark.parametrize("reason", ["timeout", "unknown_video_duration", "invalid_video_timing"])
async def test_source_timing_unavailable_fails_before_render(
    sources, tmp_path, monkeypatch, reason,
):
    async def probe(*args, **kwargs):
        raise MetadataProbeError(reason)

    async def unexpected(*args, **kwargs):
        raise AssertionError("Must not render without reliable video timing")

    monkeypatch.setattr(ffmpeg, "probe_video_timeline", probe)
    monkeypatch.setattr(ffmpeg.runner, "run", unexpected)
    with pytest.raises(ValidationError, match="reliable source video timing"):
        await ffmpeg.video_split(str(sources["long.mp4"]), str(tmp_path / "seg%03d.mp4"), [1])
    assert not list(tmp_path.iterdir())


@media
async def test_video_end_is_not_container_or_audio_end(sources, tmp_path):
    source = sources["audio-longer.mp4"]
    assert float(info(source)["format"]["duration"]) == pytest.approx(8)
    with pytest.raises(ValidationError, match="before the end"):
        await ffmpeg.video_split(str(source), str(tmp_path / "seg%03d.mp4"), [5])
    assert not list(tmp_path.iterdir())


async def upload(client, make_project, source: Path) -> tuple[str, str]:
    pid = await make_project("Split")
    response = await client.post(f"/api/v1/projects/{pid}/assets", files={
        "file": (source.name, source.read_bytes(), "video/webm" if source.suffix == ".webm"
                 else "video/mp4"),
    })
    assert response.status_code == 201, response.text
    return pid, response.json()["id"]


@media
@pytest.mark.parametrize("name,markers", [
    ("long.mp4", [1]), ("friendly.mp4", [1.3]), ("long.webm", [1]),
])
async def test_job_publishes_only_correct_split(client, make_project, sources, name, markers):
    pid, aid = await upload(client, make_project, sources[name])
    response = await client.post("/api/v1/jobs", json={
        "capability_id": "video.split", "asset_id": aid, "input": {"markers": markers},
    })
    assert response.status_code == 201, response.text
    job = (await client.get("/api/v1/jobs/" + response.json()["id"])).json()
    assert job["status"] == "succeeded", job
    assert job["output"]["segments"] == 2
    descriptors = job["output"]["outputs"]
    async with async_session_factory() as session:
        assets = list(await session.scalars(select(Asset).where(
            Asset.project_id == uuid.UUID(pid), Asset.source_asset_id == uuid.UUID(aid),
        )))
        assert {str(asset.id) for asset in assets} == {out["asset_id"] for out in descriptors}
        first = next(asset for asset in assets if str(asset.id) == descriptors[0]["asset_id"])
        timeline = await probe_video_timeline(get_storage().get_path(first.storage_path))
        assert timeline.duration_seconds == pytest.approx(markers[0], abs=0.042)


@media
@pytest.mark.parametrize("markers", [[0], [-1], [1, 1], [2, 1], [4], [5], [3.999]])
async def test_invalid_job_never_publishes_assets(client, make_project, sources, markers):
    pid, aid = await upload(client, make_project, sources["long.mp4"])
    response = await client.post("/api/v1/jobs", json={
        "capability_id": "video.split", "asset_id": aid, "input": {"markers": markers},
    })
    if markers in ([0], [-1], [1, 1]):
        assert response.status_code == 422, response.text
    else:
        assert response.status_code == 201, response.text
        job = (await client.get("/api/v1/jobs/" + response.json()["id"])).json()
        assert job["status"] == "failed", job
        assert job["output"] is None
        assert "ffmpeg" not in job["error"]
    async with async_session_factory() as session:
        assets = list(await session.scalars(
            select(Asset).where(Asset.project_id == uuid.UUID(pid))
        ))
        assert [str(asset.id) for asset in assets] == [aid]


@media
async def test_job_semantically_bad_output_publishes_nothing(client, make_project, sources,
                                                           monkeypatch):
    pid, aid = await upload(client, make_project, sources["long.mp4"])

    async def wrong_output(args, **kwargs):
        shutil.copyfile(sources["long.mp4"], args[-1])
    monkeypatch.setattr(ffmpeg.runner, "run", wrong_output)
    response = await client.post("/api/v1/jobs", json={
        "capability_id": "video.split", "asset_id": aid, "input": {"markers": [1]},
    })
    job = (await client.get("/api/v1/jobs/" + response.json()["id"])).json()
    assert job["status"] == "failed"
    assert job["output"] is None
    assert "requested boundary" in job["error"]
    async with async_session_factory() as session:
        assets = list(await session.scalars(
            select(Asset).where(Asset.project_id == uuid.UUID(pid))
        ))
        assert [str(asset.id) for asset in assets] == [aid]


async def test_public_schema_and_agent_share_split_contract(client):
    response = await client.get("/api/v1/capabilities/video.split")
    assert response.status_code == 200
    capability = response.json()
    markers = capability["input_schema"]["properties"]["markers"]
    assert markers["items"]["exclusiveMinimum"] == 0
    assert markers["minItems"] == 1
    assert markers["uniqueItems"] is True
    assert markers["x-unit"] == "seconds"
    agent = next(c for c in build_catalog() if c["capability_id"] == "video.split")
    assert agent["input_schema"] == capability["input_schema"]
    assert "does not choose which segment to keep" in agent["description"]
    assert "frame-resolution" in agent["description"]
    for value in ([0], [-1], [1, 1]):
        with pytest.raises(CapabilityInputValidationError):
            validate_capability_input(registry.get("video.split"), {"markers": value})
