"""Audio processing keeps the source's audio format (devfixes C-6, D-01).

loudnorm resamples to 192 kHz and arnndn to 48 kHz; without explicit output
settings a 16-bit 44.1 kHz mono WAV came back about 4.4 times larger, and a
video's 64k soundtrack was re-encoded at 192k.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.agent.reuse import REUSE_VERSIONS
from app.tools import ffmpeg
from app.tools.ffmpeg import AudioFormat, preserved_audio_args

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_RNNOISE = Path(ffmpeg.DEFAULT_RNNOISE_MODEL).is_file()

OPERATIONS = [
    ("audio.normalize", {}),
    ("audio.gain", {"gain_db": -3}),
    ("audio.fade", {"fade_in": 1, "fade_out": 1}),
    pytest.param(
        "audio.denoise", {},
        marks=pytest.mark.skipif(not HAVE_RNNOISE, reason="RNNoise model not installed"),
    ),
]


# --- pure argument policy -----------------------------------------------------


def test_reuse_versions_bumped() -> None:
    for capability in ("audio.gain", "audio.normalize", "audio.fade", "audio.denoise"):
        assert REUSE_VERSIONS[capability] == "2"


def test_lossless_sources_keep_codec_rate_and_layout() -> None:
    assert preserved_audio_args(
        AudioFormat(codec="pcm_s16le", sample_rate=44100, channels=1, sample_fmt="s16")
    ) == ["-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1"]
    assert preserved_audio_args(
        AudioFormat(codec="flac", sample_rate=48000, channels=2, sample_fmt="s16")
    ) == ["-c:a", "flac", "-sample_fmt", "s16", "-ar", "48000", "-ac", "2"]


def test_lossy_sources_keep_encoder_and_bitrate() -> None:
    assert preserved_audio_args(
        AudioFormat(codec="mp3", sample_rate=22050, channels=1, bit_rate=64000)
    ) == ["-c:a", "libmp3lame", "-b:a", "64000", "-ar", "22050", "-ac", "1"]
    # Opus only encodes at its own rates.
    assert preserved_audio_args(
        AudioFormat(codec="opus", sample_rate=44100, channels=2)
    ) == ["-c:a", "libopus", "-ar", "48000", "-ac", "2"]


def test_unknown_format_adds_nothing_unsafe() -> None:
    assert preserved_audio_args(AudioFormat()) == []
    assert preserved_audio_args(AudioFormat(codec="weird", sample_rate=8000)) == [
        "-ar", "8000",
    ]


# --- real FFmpeg through Jobs ---------------------------------------------------


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True,
                   capture_output=True)


def _audio_stream(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels,sample_fmt,bit_rate", "-of", "json",
         str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    stream: dict = json.loads(out)["streams"][0]
    return stream


def _video_md5(path: Path) -> str:
    return subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy",
         "-f", "md5", "-"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


async def _run(
    client: AsyncClient, pid: str, src: Path, mime: str, capability: str, params: dict,
    out_dir: Path,
) -> Path:
    resp = await client.post(
        f"{API}/projects/{pid}/assets", files={"file": (src.name, src.read_bytes(), mime)}
    )
    assert resp.status_code == 201, resp.text
    job = await client.post(f"{API}/jobs", json={
        "capability_id": capability, "asset_id": resp.json()["id"], "input": params,
    })
    assert job.status_code == 201, job.text
    body = (await client.get(f"{API}/jobs/{job.json()['id']}")).json()
    assert body["status"] == "succeeded", body
    output = body["output"]["outputs"][0]
    data = (await client.get(output["download_path"])).content
    path = out_dir / output["filename"]
    path.write_bytes(data)
    return path


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize(("capability", "params"), OPERATIONS)
async def test_mono_16bit_wav_keeps_its_format_and_size(
    client: AsyncClient, make_project, tmp_path: Path, capability: str, params: dict
) -> None:
    src = tmp_path / "speech.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=10", "-ac", "1",
            "-ar", "44100", "-c:a", "pcm_s16le", str(src))
    pid = await make_project(f"fmt-{capability}")
    out = await _run(client, pid, src, "audio/wav", capability, params, tmp_path)
    stream = _audio_stream(out)
    assert (stream["codec_name"], stream["sample_rate"], stream["channels"]) == (
        "pcm_s16le", "44100", 1,
    )
    assert 0.95 < out.stat().st_size / src.stat().st_size < 1.05


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_mp3_keeps_bitrate_and_flac_keeps_bit_depth(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid = await make_project("fmt-lossy")
    mp3 = tmp_path / "voice.mp3"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=10", "-ac", "1",
            "-ar", "22050", "-c:a", "libmp3lame", "-b:a", "64k", str(mp3))
    out = await _run(client, pid, mp3, "audio/mpeg", "audio.normalize", {}, tmp_path)
    stream = _audio_stream(out)
    assert (stream["codec_name"], stream["sample_rate"], stream["channels"]) == (
        "mp3", "22050", 1,
    )
    assert int(stream["bit_rate"]) == 64000
    assert out.stat().st_size / mp3.stat().st_size < 1.1

    flac = tmp_path / "master.flac"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=10", "-ac", "2",
            "-ar", "48000", "-sample_fmt", "s16", "-c:a", "flac", str(flac))
    out = await _run(client, pid, flac, "audio/flac", "audio.normalize", {}, tmp_path)
    stream = _audio_stream(out)
    assert (stream["codec_name"], stream["sample_rate"], stream["channels"],
            stream["sample_fmt"]) == ("flac", "48000", 2, "s16")


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize(("capability", "params"), OPERATIONS)
async def test_video_soundtrack_keeps_its_bitrate_and_picture_is_copied(
    client: AsyncClient, make_project, tmp_path: Path, capability: str, params: dict
) -> None:
    src = tmp_path / "talk.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=10", "-ac", "1",
            "-ar", "44100", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-b:a", "64k", "-shortest", str(src))
    pid = await make_project(f"fmt-video-{capability}")
    out = await _run(client, pid, src, "video/mp4", capability, params, tmp_path)
    # D-01: the picture is stream-copied, bit for bit.
    assert _video_md5(out) == _video_md5(src)
    stream = _audio_stream(out)
    assert (stream["codec_name"], stream["sample_rate"], stream["channels"]) == (
        "aac", "44100", 1,
    )
    assert int(stream["bit_rate"]) < 80_000
    assert out.stat().st_size / src.stat().st_size < 1.15
