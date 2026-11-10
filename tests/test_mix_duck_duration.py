"""audio.mix mix/duck: silent videos keep their length; cover art is never copied."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

from app.tools import ffmpeg
from app.tools.compose_policy import DEFAULT_COMPOSE_RENDER_POLICY, ComposeRenderPolicy
from tests.test_compose_soundtrack import _amplitude, _samples
from tests.test_composition_audio_clips import _tone, _upload, _video
from tests.test_composition_clip_timing import NEED_FFMPEG
from tests.test_replace_mode_duration import (
    POLICIES,
    _audio_end,
    _measure,
    _song_with_cover,
    _streams,
    _video_with_cover,
)

MODES = ["mix", "duck"]


async def _mix(tmp_path: Path, media: Path, music: Path, *, mode: str, volume: float = 0.6,
               policy: ComposeRenderPolicy | None = None, ext: str = "mp4") -> Path:
    suffix = policy.suffix if policy is not None else f".{ext}"
    out = tmp_path / f"out-{mode}-{volume}{suffix}"
    await ffmpeg.audio_mix(str(media), str(music), str(out), music_volume=volume, mode=mode,
                           apply_replace_gain=True, compose_policy=policy)
    return out


def _peak(path: Path, start: float, length: float) -> float:
    return max(abs(x) for x in _samples(path, start, length))


# --- a real video without its own sound ------------------------------------------


@pytest.mark.parametrize("policy", POLICIES, ids=["aac", "flac"])
@pytest.mark.parametrize("mode", MODES)
@NEED_FFMPEG
async def test_short_music_keeps_a_silent_video_whole(
    tmp_path: Path, mode: str, policy: ComposeRenderPolicy
) -> None:
    video = _video(tmp_path / "silent.mp4", 3, frequency=None)      # 90 frames, no audio
    music = _tone(tmp_path / "music.wav", 880, 1)
    out = await _mix(tmp_path, video, music, mode=mode, policy=policy)
    measured = _measure(out)
    assert measured["frames"] == 90                    # used to be 32: cut at 1 s
    assert _audio_end(out) == pytest.approx(3.0, abs=0.05)
    codec = "aac" if policy is DEFAULT_COMPOSE_RENDER_POLICY else "flac"
    assert [(kind, name) for kind, name, _pic in _streams(out)] == [
        ("video", "h264"), ("audio", codec)
    ]
    assert _amplitude(_samples(out, 0.2, 0.5), 880) > 0.01           # music first
    assert _peak(out, 1.3, 1.5) < 0.001                              # then silence


@pytest.mark.parametrize("mode", MODES)
@NEED_FFMPEG
async def test_long_music_is_trimmed_at_the_end_of_a_silent_video(
    tmp_path: Path, mode: str
) -> None:
    video = _video(tmp_path / "silent.mp4", 3, frequency=None)
    out = await _mix(tmp_path, video, _tone(tmp_path / "music.wav", 880, 5), mode=mode)
    assert _measure(out)["frames"] == 90
    assert _audio_end(out) == pytest.approx(3.0, abs=0.05)
    assert _amplitude(_samples(out, 2.4, 0.4), 880) > 0.01           # heard to the end


@NEED_FFMPEG
async def test_zero_volume_keeps_a_silent_video_whole(tmp_path: Path) -> None:
    video = _video(tmp_path / "silent.mp4", 3, frequency=None)
    out = await _mix(tmp_path, video, _tone(tmp_path / "music.wav", 880, 1), mode="mix",
                     volume=0.0)
    assert _measure(out)["frames"] == 90
    assert _audio_end(out) == pytest.approx(3.0, abs=0.05)
    assert _peak(out, 0.0, 3.0) < 0.001


@pytest.mark.parametrize(("mode", "volume", "expected"), [
    ("mix", 0.6, 0.6), ("mix", 0.1, 0.1), ("duck", 0.6, 0.2), ("duck", 0.1, 0.1),
])
@NEED_FFMPEG
async def test_music_volume_and_duck_cap_on_a_silent_video(
    tmp_path: Path, mode: str, volume: float, expected: float
) -> None:
    video = _video(tmp_path / "silent.mp4", 3, frequency=None)
    music = _tone(tmp_path / "music.wav", 880, 5)
    out = await _mix(tmp_path, video, music, mode=mode, volume=volume)
    ratio = (_amplitude(_samples(out, 0.5, 1.0), 880)
             / _amplitude(_samples(music, 0.5, 1.0), 880))
    assert ratio == pytest.approx(expected, rel=0.06)   # no original audio: not halved


@pytest.mark.parametrize(("mode", "music_ratio"), [("mix", 0.3), ("duck", 0.1)])
@NEED_FFMPEG
async def test_video_with_sound_is_mixed_as_before(
    tmp_path: Path, mode: str, music_ratio: float
) -> None:
    video = _video(tmp_path / "sound.mp4", 3, frequency=440)
    music = _tone(tmp_path / "music.wav", 880, 5)
    out = await _mix(tmp_path, video, music, mode=mode, volume=0.6)
    assert _measure(out)["frames"] == 90
    assert _audio_end(out) == pytest.approx(3.0, abs=0.05)
    window = (0.5, 1.0)
    source = _amplitude(_samples(video, *window), 440)
    # amix halves both inputs, as before: source x0.5, music x(volume or 0.2)/2.
    assert _amplitude(_samples(out, *window), 440) / source == pytest.approx(0.5, rel=0.08)
    assert (_amplitude(_samples(out, *window), 880)
            / _amplitude(_samples(music, *window), 880)) == pytest.approx(music_ratio, rel=0.08)


# --- embedded cover art ------------------------------------------------------------


@pytest.mark.parametrize("ext", ["mp3", "m4a"])
@pytest.mark.parametrize("mode", MODES)
@NEED_FFMPEG
async def test_song_with_cover_is_mixed_as_audio(tmp_path: Path, ext: str, mode: str) -> None:
    song = _song_with_cover(tmp_path, ext)                           # 2 s of 440 Hz
    out = await _mix(tmp_path, song, _tone(tmp_path / "music.wav", 880, 1), mode=mode,
                     ext="m4a")
    assert _streams(out) == [("audio", "aac", 0)]                    # cover not copied
    assert _audio_end(out) == pytest.approx(2.0, abs=0.06)           # used to be ~0.08 s
    assert _amplitude(_samples(out, 0.2, 0.5), 440) > 0.01           # original audio
    assert _amplitude(_samples(out, 0.2, 0.5), 880) > 0.005          # added music
    assert _amplitude(_samples(out, 1.3, 0.5), 440) > 0.01


@pytest.mark.parametrize("policy", POLICIES, ids=["aac", "flac"])
@pytest.mark.parametrize("mode", MODES)
@NEED_FFMPEG
async def test_video_with_cover_keeps_its_sound(
    tmp_path: Path, mode: str, policy: ComposeRenderPolicy
) -> None:
    video = _video_with_cover(tmp_path)                              # 90 frames, 440 Hz
    out = await _mix(tmp_path, video, _tone(tmp_path / "music.wav", 880, 1), mode=mode,
                     policy=policy)
    codec = "aac" if policy is DEFAULT_COMPOSE_RENDER_POLICY else "flac"
    assert _streams(out) == [("video", "h264", 0), ("audio", codec, 0)]
    assert _measure(out)["frames"] == 90
    assert _audio_end(out) == pytest.approx(3.0, abs=0.05)           # used to be ~0.09 s
    assert _amplitude(_samples(out, 0.2, 0.5), 880) > 0.005
    assert _amplitude(_samples(out, 2.0, 0.5), 440) > 0.01


# --- the public capability -----------------------------------------------------------


async def _job(client: AsyncClient, pid: str, media: str, music: str) -> dict:
    response = await client.post("/api/v1/jobs", json={
        "capability_id": "audio.mix", "project_id": pid, "asset_id": media,
        "input": {"music_asset_id": music, "mode": "mix", "music_volume": 0.6},
    })
    assert response.status_code == 201, response.text
    return (await client.get(f"/api/v1/jobs/{response.json()['id']}")).json()


@NEED_FFMPEG
async def test_audio_mix_job_on_a_silent_video(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid = await make_project("mix-silent-video")
    video = await _upload(client, pid, "silent.mp4",
                          _video(tmp_path / "silent.mp4", 3, frequency=None).read_bytes())
    music = await _upload(client, pid, "music.wav",
                          _tone(tmp_path / "music.wav", 880, 1).read_bytes())
    job = await _job(client, pid, video, music)
    assert job["status"] == "succeeded", job.get("error")
    output = tmp_path / "mixed.mp4"
    output.write_bytes((await client.get(job["output"]["outputs"][0]["download_path"])).content)
    assert _measure(output)["frames"] == 90
    assert _audio_end(output) == pytest.approx(3.0, abs=0.05)


@NEED_FFMPEG
async def test_unreadable_music_creates_no_asset(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid = await make_project("mix-broken-music")
    video = await _upload(client, pid, "silent.mp4",
                          _video(tmp_path / "silent.mp4", 3, frequency=None).read_bytes())
    music = await _upload(client, pid, "music.wav", b"not audio")
    before = {a["id"] for a in (await client.get(f"/api/v1/projects/{pid}/assets")).json()}
    job = await _job(client, pid, video, music)
    assert job["status"] == "failed" and job["output"] is None
    after = {a["id"] for a in (await client.get(f"/api/v1/projects/{pid}/assets")).json()}
    assert after == before
