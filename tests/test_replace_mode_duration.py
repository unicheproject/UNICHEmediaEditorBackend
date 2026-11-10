"""A short `replace` bed must not shorten the video (it used to, through -shortest)."""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.core.database import async_session_factory
from app.models.asset import Asset
from app.tools import ffmpeg
from app.tools.compose_policy import (
    DEFAULT_COMPOSE_RENDER_POLICY,
    LOSSLESS_COMPOSE_RENDER_POLICY,
    ComposeRenderPolicy,
)
from app.tools.ffmpeg import PositionedAudio
from tests.test_compose_soundtrack import _amplitude, _samples
from tests.test_composition_audio_clips import (
    _document,
    _queue_export,
    _tone,
    _upload,
    _video,
)
from tests.test_composition_clip_timing import NEED_FFMPEG, _put

POLICIES = [DEFAULT_COMPOSE_RENDER_POLICY, LOSSLESS_COMPOSE_RENDER_POLICY]


def _measure(path: Path) -> dict[str, float]:
    """Decoded video frames and the audio stream's own decoded length."""
    frames = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    packets = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "packet=pts_time,duration_time", "-of", "json", str(path)],
        check=True, capture_output=True, timeout=30,
    ).stdout)["packets"]
    audio_end = max(float(p["pts_time"]) + float(p.get("duration_time") or 0) for p in packets)
    codecs = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0",
         str(path)], check=True, capture_output=True, text=True, timeout=30,
    ).stdout.split()
    return {"frames": int(frames), "audio_end": audio_end, "codecs": " ".join(sorted(codecs))}


async def _compose(tmp_path: Path, video: Path, bed: Path | None, *, mode: str = "replace",
                   policy: ComposeRenderPolicy = DEFAULT_COMPOSE_RENDER_POLICY,
                   clips: list[PositionedAudio] | None = None, name: str = "out") -> Path:
    out = tmp_path / f"{name}{policy.suffix}"
    work = tmp_path / f"work-{name}"
    work.mkdir()
    await ffmpeg.compose([str(video)], str(out), width=96, height=64, work_dir=work,
                         audio_path=str(bed) if bed else None, audio_mode=mode,
                         music_volume=0.6, policy=policy, audio_clips=clips)
    return out


@pytest.mark.parametrize("policy", POLICIES, ids=["aac", "flac"])
@NEED_FFMPEG
async def test_short_replace_bed_keeps_the_whole_video(
    tmp_path: Path, policy: ComposeRenderPolicy
) -> None:
    video = _video(tmp_path / "v.mp4", 3, frequency=440)          # 3 s at 30 fps
    bed = _tone(tmp_path / "bed.wav", 1200, 0.5)
    out = await _compose(tmp_path, video, bed, policy=policy)
    measured = _measure(out)
    assert measured["frames"] == 90
    assert measured["audio_end"] == pytest.approx(3.0, abs=0.05)
    assert measured["codecs"] == ("aac h264" if policy is DEFAULT_COMPOSE_RENDER_POLICY
                                  else "ffv1 flac")
    assert _amplitude(_samples(out, 0.1, 0.3), 1200) > 0.05       # replacement plays
    assert _amplitude(_samples(out, 0.1, 0.3), 440) < 0.002       # soundtrack removed
    late = _samples(out, 1.0, 1.5)
    assert max(abs(x) for x in late) < 0.001                       # then silence


@pytest.mark.parametrize(("bed_seconds", "label"), [(3, "equal"), (5, "longer")])
@NEED_FFMPEG
async def test_equal_or_longer_replace_bed_ends_with_the_video(
    tmp_path: Path, bed_seconds: float, label: str
) -> None:
    video = _video(tmp_path / "v.mp4", 3, frequency=440)
    bed = _tone(tmp_path / "bed.wav", 1200, bed_seconds)
    out = await _compose(tmp_path, video, bed, name=label)
    measured = _measure(out)
    assert measured["frames"] == 90
    assert measured["audio_end"] == pytest.approx(3.0, abs=0.05)
    assert _amplitude(_samples(out, 2.4, 0.4), 1200) > 0.05       # heard to the end


@pytest.mark.parametrize("mode", ["mix", "duck"])
@NEED_FFMPEG
async def test_mix_and_duck_still_end_with_the_video(tmp_path: Path, mode: str) -> None:
    video = _video(tmp_path / "v.mp4", 3, frequency=440)
    bed = _tone(tmp_path / "bed.wav", 1200, 0.5)
    out = await _compose(tmp_path, video, bed, mode=mode, name=mode)
    measured = _measure(out)
    assert measured["frames"] == 90
    assert measured["audio_end"] == pytest.approx(3.0, abs=0.05)
    assert _amplitude(_samples(out, 1.5, 0.5), 440) > 0.02         # soundtrack kept


@NEED_FFMPEG
async def test_replace_bed_with_positioned_audio_clips(tmp_path: Path) -> None:
    video = _video(tmp_path / "v.mp4", 3, frequency=440)
    bed = _tone(tmp_path / "bed.wav", 1200, 0.5)
    voice = _tone(tmp_path / "voice.wav", 1700, 0.5)
    out = await _compose(tmp_path, video, bed,
                         clips=[PositionedAudio(str(voice), 2.0)], name="clips")
    measured = _measure(out)
    assert measured["frames"] == 90
    assert measured["audio_end"] == pytest.approx(3.0, abs=0.05)
    assert _amplitude(_samples(out, 2.1, 0.3), 1700) > 0.05


async def _project_media(client: AsyncClient, make_project, name: str, tmp_path: Path,
                         bed_bytes: bytes) -> tuple[str, str, str]:
    pid = await make_project(name)
    video = await _upload(client, pid, "clip.mp4",
                          _video(tmp_path / "v.mp4", 3, frequency=440).read_bytes())
    bed = await _upload(client, pid, "bed.wav", bed_bytes)
    return pid, video, bed


@NEED_FFMPEG
async def test_direct_video_compose_job_keeps_the_video(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid, video, bed = await _project_media(
        client, make_project, "replace-direct", tmp_path,
        _tone(tmp_path / "bed.wav", 1200, 0.5).read_bytes(),
    )
    response = await client.post("/api/v1/jobs", json={
        "capability_id": "video.compose", "project_id": pid,
        "input": {"asset_ids": [video], "width": 96, "height": 64,
                  "audio_asset_id": bed, "audio_mode": "replace"},
    })
    assert response.status_code == 201, response.text
    job = (await client.get(f"/api/v1/jobs/{response.json()['id']}")).json()
    assert job["status"] == "succeeded", job.get("error")
    output = tmp_path / "direct.mp4"
    output.write_bytes((await client.get(job["output"]["outputs"][0]["download_path"])).content)
    measured = _measure(output)
    assert measured["frames"] == 90
    assert measured["audio_end"] == pytest.approx(3.0, abs=0.05)


@NEED_FFMPEG
async def test_project_export_keeps_the_video(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid, video, bed = await _project_media(
        client, make_project, "replace-export", tmp_path,
        _tone(tmp_path / "bed.wav", 1200, 0.5).read_bytes(),
    )
    saved = await _put(client, pid, _document(
        [{"id": str(uuid.uuid4()), "asset_id": video}],
        bed={"asset_id": bed, "mode": "replace", "volume": 0.5},
    ))
    assert saved.status_code == 201, saved.text
    job = await _queue_export(client, pid, saved.json()["revision_id"])
    from app.services.jobs import execute_job

    async with async_session_factory() as session:
        executed = await execute_job(session, uuid.UUID(job["id"]))
        assert executed.status.value == "succeeded", executed.error
        assert executed.output is not None
        asset = await session.get(Asset, uuid.UUID(executed.output["outputs"][0]["asset_id"]))
        assert asset is not None and asset.duration_seconds is not None
        assert asset.duration_seconds == pytest.approx(3.0, abs=0.1)


@NEED_FFMPEG
async def test_unreadable_replacement_creates_no_asset(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid, video, bed = await _project_media(client, make_project, "replace-broken", tmp_path,
                                           b"not audio")
    saved = await _put(client, pid, _document(
        [{"id": str(uuid.uuid4()), "asset_id": video}],
        bed={"asset_id": bed, "mode": "replace", "volume": 0.5},
    ))
    assert saved.status_code == 201, saved.text
    job = await _queue_export(client, pid, saved.json()["revision_id"])
    before = {a["id"] for a in (await client.get(f"/api/v1/projects/{pid}/assets")).json()}
    from app.services.jobs import execute_job

    async with async_session_factory() as session:
        executed = await execute_job(session, uuid.UUID(job["id"]))
    assert executed.status.value == "failed" and executed.output is None
    after = {a["id"] for a in (await client.get(f"/api/v1/projects/{pid}/assets")).json()}
    assert after == before


# --- embedded cover art (attached pictures) -------------------------------------


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True,
                   timeout=30)


def _cover(tmp_path: Path) -> Path:
    cover = tmp_path / "cover.png"
    _ff("-f", "lavfi", "-i", "color=red:s=64x64:d=0.04", "-frames:v", "1", str(cover))
    return cover


def _song_with_cover(tmp_path: Path, ext: str) -> Path:
    """2 s of 440 Hz with an attached cover picture, as music files often have."""
    song = tmp_path / f"song.{ext}"
    audio, picture = ("libmp3lame", "mjpeg") if ext == "mp3" else ("aac", "png")
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-i", str(_cover(tmp_path)),
        "-map", "0:a", "-map", "1:v", "-c:a", audio, "-c:v", picture,
        "-disposition:v:0", "attached_pic", str(song))
    return song


def _streams(path: Path) -> list[tuple[str, str, int]]:
    data = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name:stream_disposition=attached_pic", "-of", "json",
         str(path)], check=True, capture_output=True, timeout=30,
    ).stdout)
    return [(s["codec_type"], s["codec_name"], s.get("disposition", {}).get("attached_pic", 0))
            for s in data["streams"]]


def _audio_end(path: Path) -> float:
    packets = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "packet=pts_time,duration_time", "-of", "json", str(path)],
        check=True, capture_output=True, timeout=30,
    ).stdout)["packets"]
    return max(float(p["pts_time"]) + float(p.get("duration_time") or 0) for p in packets)


@pytest.mark.parametrize("ext", ["mp3", "m4a"])
@pytest.mark.parametrize("replacement_seconds", [1, 4])
@NEED_FFMPEG
async def test_cover_art_audio_is_replaced_as_audio(
    tmp_path: Path, ext: str, replacement_seconds: float
) -> None:
    song = _song_with_cover(tmp_path, ext)
    assert not await ffmpeg._has_video(str(song))          # cover art is not video
    music = _tone(tmp_path / "music.wav", 880, replacement_seconds)
    out = tmp_path / "out.m4a"
    await ffmpeg.audio_mix(str(song), str(music), str(out), music_volume=1.0,
                           mode="replace", apply_replace_gain=True)
    assert _streams(out) == [("audio", "aac", 0)]           # no cover kept as video
    # The replacement is the whole result, as long as it is (it used to end
    # after about 0.09 s, because -shortest also counted the one-frame cover).
    assert _audio_end(out) == pytest.approx(replacement_seconds, abs=0.05)
    assert _amplitude(_samples(out, 0.2, 0.5), 880) > 0.05
    assert _amplitude(_samples(out, 0.2, 0.5), 440) < 0.002


def _video_with_cover(tmp_path: Path) -> Path:
    """3 s of real video (90 frames) + 440 Hz, with an attached cover picture.

    The MP4 muxer always stores the cover after the real video whatever the
    map order (MOV drops it; MKV keeps a picture first but as an ordinary
    video stream, not an attached picture), so a cover *before* the real video
    cannot be built locally. The replace path still measures "V:0" (see
    test_audio_mix_gain), which skips attached pictures in any position.
    """
    video = tmp_path / "with-cover.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=96x64:r=30:d=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-i", str(_cover(tmp_path)),
        "-map", "0:v", "-map", "2:v", "-map", "1:a", "-c:v:0", "libx264", "-c:v:1", "png",
        "-pix_fmt", "yuv420p", "-disposition:v:1", "attached_pic", "-c:a", "aac", str(video))
    return video


@pytest.mark.parametrize("policy", POLICIES, ids=["aac", "flac"])
@NEED_FFMPEG
async def test_video_with_cover_keeps_its_real_frames(
    tmp_path: Path, policy: ComposeRenderPolicy
) -> None:
    video = _video_with_cover(tmp_path)
    assert [kind for kind, _codec, pic in _streams(video) if kind == "video" and pic] == ["video"]
    music = _tone(tmp_path / "bed.wav", 1200, 0.5)
    out = tmp_path / f"out{policy.suffix}"
    await ffmpeg.audio_mix(str(video), str(music), str(out), music_volume=0.6,
                           mode="replace", apply_replace_gain=False, compose_policy=policy)
    audio_codec = "aac" if policy is DEFAULT_COMPOSE_RENDER_POLICY else "flac"
    assert _streams(out) == [("video", "h264", 0), ("audio", audio_codec, 0)]  # cover dropped
    measured = _measure(out)
    assert measured["frames"] == 90                              # every real frame kept
    assert _audio_end(out) == pytest.approx(3.0, abs=0.05)
    assert _amplitude(_samples(out, 0.1, 0.3), 1200) > 0.05
    assert max(abs(x) for x in _samples(out, 1.0, 1.5)) < 0.001


@NEED_FFMPEG
async def test_audio_mix_job_on_a_song_with_cover(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid = await make_project("replace-cover-art")
    song = await _upload(client, pid, "song.mp3", _song_with_cover(tmp_path, "mp3").read_bytes())
    music = await _upload(client, pid, "music.wav", _tone(tmp_path / "m.wav", 880, 1).read_bytes())
    response = await client.post("/api/v1/jobs", json={
        "capability_id": "audio.mix", "project_id": pid, "asset_id": song,
        "input": {"music_asset_id": music, "mode": "replace", "music_volume": 1},
    })
    assert response.status_code == 201, response.text
    job = (await client.get(f"/api/v1/jobs/{response.json()['id']}")).json()
    assert job["status"] == "succeeded", job.get("error")
    output = tmp_path / "mixed.m4a"
    output.write_bytes((await client.get(job["output"]["outputs"][0]["download_path"])).content)
    assert _streams(output) == [("audio", "aac", 0)]
    assert _audio_end(output) == pytest.approx(1.0, abs=0.05)
