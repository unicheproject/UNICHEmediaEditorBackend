"""Real signal regressions for shared composition soundtrack semantics."""

from __future__ import annotations

import array
import json
import math
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.core.database import async_session_factory
from app.core.errors import ToolError
from app.models.asset import Asset
from app.services.jobs import execute_job
from app.services.storage import get_storage
from app.tools import ffmpeg
from app.tools.compose_policy import (
    DEFAULT_COMPOSE_RENDER_POLICY,
    LOSSLESS_COMPOSE_RENDER_POLICY,
)
from tests.test_project_export import _document, _queue_export, _save

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs media tools"
)
POLICIES = [DEFAULT_COMPOSE_RENDER_POLICY, LOSSLESS_COMPOSE_RENDER_POLICY]


def _run(args: list[str]) -> bytes:
    return subprocess.run(args, check=True, capture_output=True, timeout=30).stdout


def _clip(
    path: Path,
    frequency: int | None,
    *,
    audio_duration: float = 1,
    sample_rate: int = 44100,
    second_frequency: int | None = None,
    audio_offset: float = 0,
) -> None:
    args = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=96x64:r=30:d=1",
    ]
    if frequency is not None:
        args += [
            "-itsoffset",
            str(audio_offset),
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:sample_rate={sample_rate}:duration={audio_duration}",
        ]
        if second_frequency is not None:
            args += [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={second_frequency}:duration=1",
            ]
        args += ["-map", "0:v:0", "-map", "1:a:0"]
        if second_frequency is not None:
            args += ["-map", "2:a:0"]
        args += ["-c:a", "flac"]
    args += ["-c:v", "ffv1", str(path)]
    _run(args)


def _samples(path: Path, start: float = 0.2, duration: float = 0.4) -> array.array:
    return array.array(
        "f",
        _run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                "8000",
                "-f",
                "f32le",
                "-",
            ]
        ),
    )


def _rms(samples: array.array) -> float:
    assert samples, "missing decoded audio"
    return math.sqrt(sum(value * value for value in samples) / len(samples))


def _amplitude(samples: array.array, frequency: int) -> float:
    # Tone projection distinguishes order and mixing without an FFT dependency.
    real = sum(
        value * math.cos(2 * math.pi * frequency * i / 8000) for i, value in enumerate(samples)
    )
    imag = sum(
        value * math.sin(2 * math.pi * frequency * i / 8000) for i, value in enumerate(samples)
    )
    return 2 * math.hypot(real, imag) / len(samples)


def _probe(path: Path) -> dict:
    return json.loads(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )


@pytest.mark.parametrize("policy", POLICIES, ids=["default", "lossless"])
@pytest.mark.parametrize("frequencies", [(440,), (440, 880, 440), (440, None, 880), (None, None)])
async def test_ordered_source_audio_and_silence(tmp_path, policy, frequencies) -> None:
    sources = []
    for index, frequency in enumerate(frequencies):
        source = tmp_path / f"source-{index}.mkv"
        _clip(source, frequency, sample_rate=48000 if index == 1 else 44100)
        sources.append(str(source))
    output = tmp_path / f"composition{policy.suffix}"
    await ffmpeg.compose(
        sources, str(output), width=96, height=64, work_dir=tmp_path, policy=policy
    )
    metadata = _probe(output)
    streams = {stream["codec_type"]: stream for stream in metadata["streams"]}
    assert streams["audio"]["codec_name"] == policy.audio_codec.value
    assert streams["audio"]["sample_rate"] == "44100"
    assert streams["audio"]["channels"] == 2
    assert abs(float(metadata["format"]["duration"]) - len(frequencies)) < 0.12
    if policy == LOSSLESS_COMPOSE_RENDER_POLICY:
        assert streams["audio"]["sample_fmt"] == "s32"
        assert streams["video"]["codec_name"] == "ffv1"
    for index, frequency in enumerate(frequencies):
        samples = _samples(output, index + 0.2)
        if frequency is None:
            assert _rms(samples) < 0.0001
        else:
            assert _rms(samples) > 0.04
            assert _amplitude(samples, frequency) > 0.06
            other = 880 if frequency == 440 else 440
            assert _amplitude(samples, other) < 0.001


@pytest.mark.parametrize("policy", POLICIES, ids=["default", "lossless"])
@pytest.mark.parametrize("audio_duration", [0.4, 2])
async def test_audio_is_padded_or_trimmed_to_video_duration(tmp_path, policy, audio_duration):
    first, second = tmp_path / "first.mkv", tmp_path / "second.mkv"
    _clip(first, 440, audio_duration=audio_duration)
    _clip(second, 880)
    output = tmp_path / f"composition{policy.suffix}"
    await ffmpeg.compose(
        [str(first), str(second)],
        str(output),
        width=96,
        height=64,
        work_dir=tmp_path,
        policy=policy,
    )
    assert abs(float(_probe(output)["format"]["duration"]) - 2) < 0.12
    assert _amplitude(_samples(output, 0.1, 0.2), 440) > 0.06
    if audio_duration < 1:
        assert _rms(_samples(output, 0.6, 0.2)) < 0.0001
    assert _amplitude(_samples(output, 1.2), 880) > 0.06


@pytest.mark.parametrize("policy", POLICIES, ids=["default", "lossless"])
@pytest.mark.parametrize("mode,bed_ratio", [("mix", 0.6), ("duck", 0.2), ("replace", None)])
async def test_audio_bed_keeps_existing_mix_duck_replace_semantics(
    tmp_path,
    policy,
    mode,
    bed_ratio,
):
    source, bed = tmp_path / "source.mkv", tmp_path / "bed.wav"
    _clip(source, 440)
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1200:sample_rate=44100:duration=2",
            str(bed),
        ]
    )
    output = tmp_path / f"composition{policy.suffix}"
    await ffmpeg.compose(
        [str(source)],
        str(output),
        width=96,
        height=64,
        work_dir=tmp_path,
        policy=policy,
        audio_path=str(bed),
        audio_mode=mode,
        music_volume=0.6,
    )
    samples = _samples(output)
    primary, music = _amplitude(samples, 440), _amplitude(samples, 1200)
    if mode == "replace":
        assert primary < 0.001
        assert music > 0.1  # replace keeps the unscaled bed, as before
    else:
        assert primary > 0.025
        # Mono source is normalized to stereo; compare to the normalized source
        # rather than assuming a volume for FFmpeg's channel-layout conversion.
        original = _samples(tmp_path / f"_norm_000{policy.suffix}")
        bed_samples = _samples(bed)
        assert primary / _amplitude(original, 440) == pytest.approx(0.5, abs=0.04)
        assert music / _amplitude(bed_samples, 1200) == pytest.approx(bed_ratio / 2, abs=0.04)


@pytest.mark.parametrize("policy", POLICIES, ids=["default", "lossless"])
async def test_only_first_source_soundtrack_is_selected(tmp_path, policy):
    source = tmp_path / "two-tracks.mkv"
    _clip(source, 440, second_frequency=880)
    output = tmp_path / f"composition{policy.suffix}"
    await ffmpeg.compose(
        [str(source)], str(output), width=96, height=64, work_dir=tmp_path, policy=policy
    )
    assert len([s for s in _probe(output)["streams"] if s["codec_type"] == "audio"]) == 1
    samples = _samples(output)
    assert _amplitude(samples, 440) > 0.06
    assert _amplitude(samples, 880) < 0.001


@pytest.mark.parametrize("policy", POLICIES, ids=["default", "lossless"])
async def test_delayed_soundtrack_retains_leading_silence(tmp_path, policy):
    source = tmp_path / "delayed.mkv"
    _clip(source, 440, audio_offset=0.3)
    output = tmp_path / f"composition{policy.suffix}"
    await ffmpeg.compose(
        [str(source)], str(output), width=96, height=64, work_dir=tmp_path, policy=policy
    )
    assert _rms(_samples(output, 0.05, 0.1)) < 0.0001
    assert _amplitude(_samples(output, 0.5, 0.2), 440) > 0.06
    assert abs(float(_probe(output)["format"]["duration"]) - 1) < 0.05


@pytest.mark.parametrize("policy", POLICIES, ids=["default", "lossless"])
@pytest.mark.parametrize("has_audio", [False, True])
async def test_normalization_command_selects_source_or_synthetic_audio(
    tmp_path,
    monkeypatch,
    policy,
    has_audio,
):
    commands = []

    async def probe(args, **kwargs):
        if "v:0" in args:
            return '{"streams":[{"duration":"1"}]}'
        return "1\n" if has_audio else ""

    async def run(args, **kwargs):
        commands.append(args)
        return ""

    monkeypatch.setattr(ffmpeg.runner, "run_stdout", probe)
    monkeypatch.setattr(ffmpeg.runner, "run", run)
    await ffmpeg._normalize_segment(
        "source.mkv", str(tmp_path / f"out{policy.suffix}"), 96, 64, policy
    )
    command = commands[0]
    assert ("anullsrc" in " ".join(command)) is not has_audio
    mappings = [command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "-map"]
    assert mappings == ["0:v:0", "0:a:0" if has_audio else "1:a:0"]
    assert command[command.index("-c:a") + 1] == policy.audio_codec.value
    assert command[command.index("-t") + 1] == "1.0"


@pytest.mark.parametrize(
    "response,duration",
    [
        ({"streams": [{"duration": "1.25"}]}, 1.25),
        ({"streams": [{"tags": {"DURATION": "00:00:01.250"}}], "format": {"duration": "9"}}, 1.25),
        ({"streams": [{}], "format": {"duration": "1.25"}}, 1.25),
    ],
)
async def test_video_duration_does_not_use_longer_audio_duration(monkeypatch, response, duration):
    async def probe(args, **kwargs):
        return json.dumps(response)

    monkeypatch.setattr(ffmpeg.runner, "run_stdout", probe)
    assert await ffmpeg._compose_video_duration("source.mkv") == duration


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "null",
        '{"streams":[]}',
        '{"streams":[{"duration":"nan"}]}',
        '{"streams":[{"duration":"0"}]}',
        '{"streams":[{"tags":{"DURATION":"bad"}}]}',
    ],
)
async def test_unknown_video_duration_fails_safely(monkeypatch, response):
    async def probe(args, **kwargs):
        return response

    monkeypatch.setattr(ffmpeg.runner, "run_stdout", probe)
    with pytest.raises(ToolError, match="video duration"):
        await ffmpeg._compose_video_duration("source.mkv")


@pytest.mark.parametrize("operation", ["video.compose", "web", "preview", "archival"])
async def test_direct_compose_and_export_jobs_retain_soundtrack(
    client: AsyncClient,
    make_project,
    tmp_path,
    operation,
):
    project_id = await make_project("Soundtrack")
    source = tmp_path / "source.mkv"
    _clip(source, 440)
    response = await client.post(
        f"/api/v1/projects/{project_id}/assets",
        files={
            "file": (source.name, source.read_bytes(), "video/x-matroska"),
        },
    )
    assert response.status_code == 201, response.text
    asset_id = response.json()["id"]
    if operation == "video.compose":
        response = await client.post(
            "/api/v1/jobs",
            json={
                "project_id": project_id,
                "capability_id": operation,
                "input": {"asset_ids": [asset_id], "width": 96, "height": 64},
            },
        )
        assert response.status_code == 201, response.text
        job_id = response.json()["id"]
    else:
        revision = await _save(client, project_id, _document([asset_id]))
        job = await _queue_export(client, project_id, revision["revision_id"], preset=operation)
        job_id = job["id"]
    async with async_session_factory() as session:
        job = await execute_job(session, uuid.UUID(job_id))
        assert job.status.value == "succeeded", job.error
        assert len(job.output["outputs"]) == 1
        output = await session.get(Asset, uuid.UUID(job.output["outputs"][0]["asset_id"]))
        assert output is not None
        output_path = get_storage().get_path(output.storage_path)
        assert _rms(_samples(output_path)) > 0.04
        assert _amplitude(_samples(output_path), 440) > 0.06
        metadata = _probe(output_path)
        codecs = {s["codec_type"]: s["codec_name"] for s in metadata["streams"]}
        assert codecs == (
            {"video": "ffv1", "audio": "flac"}
            if operation == "archival"
            else {"video": "h264", "audio": "aac"}
        )
