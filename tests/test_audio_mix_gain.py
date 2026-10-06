"""Public audio.mix gain semantics, measured through real Job publication."""

from __future__ import annotations

import array
import math
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.core.database import async_session_factory
from app.core.errors import ToolError
from app.models.asset import Asset
from app.tools import ffmpeg
from tests.test_compose_capabilities import _project, _run_job, _upload

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs media tools"
)


def _run(args: list[str]) -> bytes:
    return subprocess.run(args, check=True, capture_output=True, timeout=30).stdout


@pytest.fixture(scope="module")
def signals(tmp_path_factory):
    root = tmp_path_factory.mktemp("mix-gain")
    for name, duration, frequency in [
        ("source.wav", 2, 440), ("music.wav", 2, 880),
        ("short.wav", 1, 880), ("long.wav", 4, 880),
    ]:
        _run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            f"sine=frequency={frequency}:sample_rate=48000:duration={duration}",
            "-c:a", "pcm_s16le", str(root / name),
        ])
    for name, audible in [("source.mp4", True), ("silent.mp4", False)]:
        args = [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=duration=2:size=96x64:rate=25",
        ]
        if audible:
            args += ["-i", str(root / "source.wav"), "-c:a", "aac"]
        _run([*args, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(root / name)])
    return root


def _samples(path: Path) -> array.array:
    # Interior half-second window: integer cycles of both test tones, no codec tails.
    return array.array("f", _run([
        "ffmpeg", "-v", "error", "-i", str(path), "-ss", "0.4", "-t", "0.5",
        "-map", "0:a:0", "-ac", "1", "-ar", "8000", "-f", "f32le", "-",
    ]))


def _tone_rms(samples: array.array, frequency: int) -> float:
    sine = sum(x * math.sin(2 * math.pi * frequency * i / 8000)
               for i, x in enumerate(samples))
    cosine = sum(x * math.cos(2 * math.pi * frequency * i / 8000)
                 for i, x in enumerate(samples))
    return math.sqrt(2) * math.hypot(sine, cosine) / len(samples)


async def _submit(client, signals, source, volume, *, mode="replace", music="music.wav"):
    pid = await _project(client)
    primary = await _upload(
        client, pid, str(signals / source), source,
        "video/mp4" if source.endswith(".mp4") else "audio/wav",
    )
    supplied = await _upload(client, pid, str(signals / music), music, "audio/wav")
    job = await _run_job(client, {
        "capability_id": "audio.mix", "project_id": pid, "asset_id": primary,
        "input": {"music_asset_id": supplied, "mode": mode, "music_volume": volume},
    })
    return job, pid, primary


async def _published(client, job, pid, primary, path):
    assert job["status"] == "succeeded", job.get("error")
    assert job["output"]["operation"] == "audio.mix"
    assert len(job["output"]["outputs"]) == 1
    output = job["output"]["outputs"][0]
    asset = (await client.get(f"/api/v1/assets/{output['asset_id']}")).json()
    assert asset["source_asset_id"] == primary
    assert asset["technical_metadata_status"] == "available"
    assert asset["audio_codec"] == "aac"
    async with async_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Asset).where(
            Asset.project_id == uuid.UUID(pid), Asset.source_asset_id.is_not(None),
        )) == 1
    response = await client.get(output["download_path"])
    assert response.status_code == 200
    path.write_bytes(response.content)
    # Both zero-gain and non-zero outputs must remain decodable media.
    _run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    return asset


@pytest.mark.parametrize("source", ["source.mp4", "silent.mp4", "source.wav"])
@pytest.mark.parametrize("volume", [1.0, 0.5, 0.1, 0.0])
async def test_replace_gain_and_source_suppression(client, signals, tmp_path, source, volume):
    job, pid, primary = await _submit(client, signals, source, volume)
    is_video = source.endswith(".mp4")
    path = tmp_path / ("mixed.mp4" if is_video else "mixed.m4a")
    asset = await _published(client, job, pid, primary, path)
    assert asset["media_type"] == ("video" if is_video else "audio")
    assert asset["extension"] == ("mp4" if is_video else "m4a")
    assert asset["has_video_stream"] is is_video
    if is_video:
        assert asset["video_codec"] == "h264"
        # Encoded video packets are unchanged, not merely the same codec.
        def video_hash(p):
            return _run(["ffmpeg", "-v", "error", "-i", str(p), "-map", "0:v:0",
                         "-c:v", "copy", "-f", "hash", "-"])
        assert video_hash(path) == video_hash(signals / source)
    samples = _samples(path)
    original, music = _tone_rms(samples, 440), _tone_rms(samples, 880)
    reference = _tone_rms(_samples(signals / "music.wav"), 880)
    assert original < 0.0001
    if volume:
        assert music / reference == pytest.approx(volume, rel=0.05)
    else:
        assert math.sqrt(sum(x*x for x in samples) / len(samples)) < 0.0001
    print(f"gain source={source} volume={volume} music_rms={music:.7f} source_rms={original:.7f}")


@pytest.mark.parametrize("source", ["source.mp4", "source.wav"])
@pytest.mark.parametrize("music,duration", [("short.wav", 1), ("long.wav", 4)])
async def test_replace_duration_unchanged(client, signals, tmp_path, source, music, duration):
    job, pid, primary = await _submit(client, signals, source, 0.5, music=music)
    path = tmp_path / ("mixed.mp4" if source.endswith(".mp4") else "mixed.m4a")
    asset = await _published(client, job, pid, primary, path)
    expected = min(2, duration) if source.endswith(".mp4") else duration
    # Stream-copy video can leave a packet/reorder tail under -shortest.
    tail_tolerance = 0.3 if source.endswith(".mp4") else 0.05
    assert expected - 0.05 <= asset["duration_seconds"] <= expected + tail_tolerance


@pytest.mark.parametrize("mode", ["mix", "duck"])
@pytest.mark.parametrize("volume", [1.0, 0.5, 0.1, 0.0])
async def test_mix_and_duck_gain_unchanged(client, signals, tmp_path, mode, volume):
    job, pid, primary = await _submit(client, signals, "source.wav", volume, mode=mode)
    path = tmp_path / "mixed.m4a"
    await _published(client, job, pid, primary, path)
    samples = _samples(path)
    reference_music = _tone_rms(_samples(signals / "music.wav"), 880)
    reference_source = _tone_rms(_samples(signals / "source.wav"), 440)
    effective = min(volume, 0.2) if mode == "duck" else volume
    assert _tone_rms(samples, 440) / reference_source == pytest.approx(0.5, rel=0.05)
    assert _tone_rms(samples, 880) / reference_music == pytest.approx(effective / 2, abs=0.005)


@pytest.mark.parametrize("mode", ["mix", "replace"])
@pytest.mark.parametrize("volume", [2.0, -1.0])
async def test_replace_numeric_values_match_mix(client, signals, tmp_path, mode, volume):
    job, pid, primary = await _submit(client, signals, "source.wav", volume, mode=mode)
    path = tmp_path / "mixed.m4a"
    await _published(client, job, pid, primary, path)
    ratio = _tone_rms(_samples(path), 880) / _tone_rms(_samples(signals / "music.wav"), 880)
    # FFmpeg accepts signed gain (negative values invert phase); no new range policy.
    expected = abs(volume) / 2 if mode == "mix" else abs(volume)
    assert ratio == pytest.approx(expected, rel=0.05)


async def test_replace_filter_failure_publishes_nothing(client, signals, monkeypatch):
    partial_outputs = []
    async def fail(args, **kwargs):
        path = Path(args[-1])
        path.write_bytes(b"partial encoding")
        partial_outputs.append(path)
        raise ToolError("Audio filter execution failed")
    monkeypatch.setattr(ffmpeg.runner, "run", fail)
    job, pid, _ = await _submit(client, signals, "source.mp4", 0.5)
    assert job["status"] == "failed"
    assert job["output"] is None
    assert job["error"] == "Audio filter execution failed"
    assert partial_outputs and all(not path.parent.exists() for path in partial_outputs)
    async with async_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Asset).where(
            Asset.project_id == uuid.UUID(pid), Asset.source_asset_id.is_not(None),
        )) == 0


@pytest.mark.parametrize("apply_gain", [True, False])
async def test_replacement_policy_is_explicit(monkeypatch, apply_gain):
    commands = []
    async def has_audio(path):
        return True
    async def capture(args, **kwargs):
        commands.append(args)
    monkeypatch.setattr(ffmpeg, "_has_audio", has_audio)
    monkeypatch.setattr(ffmpeg.runner, "run", capture)
    await ffmpeg.audio_mix("source", "music", "output.mp4", music_volume=0.5,
                           mode="replace", apply_replace_gain=apply_gain)
    command = commands[0]
    assert "-shortest" in command
    assert command[command.index("-c:v") + 1] == "copy"
    if apply_gain:
        assert command[command.index("-filter_complex") + 1] == "[1:a]volume=0.5[a]"
        assert "[a]" in command
        assert "1:a" not in command
    else:
        assert "-filter_complex" not in command
        assert "1:a" in command
