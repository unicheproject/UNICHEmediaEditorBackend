"""Task #18B: FFmpeg/FFprobe may only open stored files with allowed demuxers.

The reproduced issue: an uploaded ".mp4" that is really an HLS playlist was
opened as HLS, and FFmpeg read the local media it referenced into the output.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.agent.render import compile_execution_plan, render_group
from app.agent.schemas import Plan
from app.core.errors import ToolError
from app.models.enums import MediaType
from app.services import media_metadata
from app.services.media_metadata import MetadataProbeError, probe_media
from app.tools import ffmpeg, process_policy, pyscenedetect, runner, waveform
from app.tools.demuxers import (
    FORMAT_WHITELIST,
    UNTRUSTED_INPUT,
    UnrestrictedInputError,
    check_input_policy,
)

API = "/api/v1"
NEED_FFMPEG = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe") and shutil.which("convert")),
    reason="FFmpeg, FFprobe and ImageMagick are required",
)
R = list(UNTRUSTED_INPUT)


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True,
                   capture_output=True, timeout=30)


def _secret_media(path: Path) -> Path:
    """A harmless red clip standing in for another project's media."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ff("-f", "lavfi", "-i", "color=c=red:s=64x48:d=1", "-c:v", "mpeg2video", str(path))
    return path


def _playlist(target: str) -> bytes:
    return (f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n{target}\n"
            "#EXT-X-ENDLIST\n").encode()


# --- the argv policy ---------------------------------------------------------------------


@pytest.mark.parametrize("args", [
    ["ffmpeg", "-y", *R, "-i", "/a.mp4", "/o.mp4"],
    ["ffmpeg", "-y", "-ss", "1", *R, "-i", "/a.mp4", *R, "-i", "/b.wav", "/o.mp4"],
    ["ffmpeg", "-y", *R, "-i", "/a.mp4", "-f", "lavfi", "-i", "anullsrc", "/o.mp4"],
    ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "/work/list.txt", "/o.mkv"],
    ["ffmpeg", "-y", *R, "-i", "/v.mp4", "-f", "srt", "-i", "/s.srt", "/o.mp4"],
    ["ffprobe", "-v", "error", "-of", "json", *R, "-i", "/a.mp4"],
    ["ffprobe", "-v", "error", "-f", "webvtt", "-i", "/s.vtt"],
    ["convert", "/a.png", "/b.png"],          # not FFmpeg: ImageMagick has its own policy
    ["identify", "-ping", "/a.png"],
])
def test_restricted_commands_pass(args: list[str]) -> None:
    check_input_policy(args)


@pytest.mark.parametrize("args", [
    ["ffmpeg", "-y", "-i", "/a.mp4", "/o.mp4"],
    # The allowlist of the first input does not cover the second one.
    ["ffmpeg", "-y", *R, "-i", "/a.mp4", "-i", "/b.wav", "/o.mp4"],
    # Options after -i belong to the next input or the output, not to this input.
    ["ffmpeg", "-y", "-i", "/a.mp4", *R, "/o.mp4"],
    # A forced format is trusted only for application-built inputs.
    ["ffmpeg", "-y", "-f", "hls", "-i", "/a.mp4", "/o.mp4"],
    ["ffmpeg", "-y", "-f", "mov", "-i", "/a.mp4", "/o.mp4"],
    # Half the allowlist (no protocol restriction, or another demuxer list).
    ["ffmpeg", "-y", "-format_whitelist", FORMAT_WHITELIST, "-i", "/a.mp4", "/o.mp4"],
    ["ffmpeg", "-y", "-protocol_whitelist", "file", "-format_whitelist", "hls", "-i", "/a",
     "/o.mp4"],
    ["ffprobe", "-v", "error", "/a.mp4"],                      # positional input
    ["ffprobe", "-v", "error", "-i", "/a.mp4"],
    ["ffprobe", "-v", "error", *R, "-i", "/a.mp4", "-i", "/b.mp4"],
    ["/usr/bin/ffmpeg", "-i", "/a.mp4", "/o.mp4"],
])
def test_unrestricted_commands_are_refused(args: list[str]) -> None:
    with pytest.raises(UnrestrictedInputError):
        check_input_policy(args)


def test_allowlist_excludes_reference_demuxers() -> None:
    allowed = set(FORMAT_WHITELIST.split(","))
    assert allowed == {"mov", "matroska", "avi", "mp3", "wav", "flac", "ogg",
                       "png_pipe", "jpeg_pipe", "webp_pipe", "tiff_pipe"}
    assert not allowed & {"hls", "applehttp", "dash", "concat", "image2", "ffmetadata",
                          "tee", "lavfi", "rtsp", "sdp", "mpd"}
    assert UNTRUSTED_INPUT[:2] == ("-protocol_whitelist", "file")


async def test_media_process_refuses_before_launching(monkeypatch) -> None:
    async def forbidden(*_args, **_kwargs):
        raise AssertionError("an unrestricted command must not be launched")

    monkeypatch.setattr(process_policy.asyncio, "create_subprocess_exec", forbidden)
    with pytest.raises(ToolError, match="^Media processing failed.$"):
        async with process_policy.media_process(
            ["ffmpeg", "-i", "/a.mp4", "/o.mp4"],
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        ):
            pass


async def test_every_input_of_the_media_tools_is_restricted(monkeypatch, tmp_path) -> None:
    """Command-level audit: each file input carries the allowlist right before -i."""
    commands: list[list[str]] = []

    async def run(args: list[str], **_kwargs) -> str:
        commands.append(list(args))
        return ""

    async def run_stdout(args: list[str], **_kwargs) -> str:
        commands.append(list(args))
        if "stream=index" in args:
            return "0\n"
        if "stream=width,height" in args:
            return '{"streams":[{"width":64,"height":48}]}'
        return '{"streams":[{"duration":"2"}],"format":{"duration":"2"}}'

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(runner, "run_stdout", run_stdout)
    user = ["/store/v.mp4", "/store/a.wav", "/store/p.png", "/store/m.mp3"]
    calls = [
        ffmpeg.video_trim(user[0], "/o.mp4", 0, 1),
        ffmpeg.video_mute(user[0], "/o.mp4"),
        ffmpeg.video_transcode(user[0], "/o.mkv", None, None),
        ffmpeg.video_audio_extract(user[0], "/o.m4a"),
        ffmpeg.overlay_lowerthird(user[0], user[2], "/o.mp4", start=0, duration=1),
        ffmpeg.audio_mix(user[0], user[3], "/o.mp4", mode="mix", apply_replace_gain=True),
        ffmpeg.audio_mix(user[1], user[3], "/o.wav", mode="replace",
                         apply_replace_gain=True),
        ffmpeg.still_to_clip(user[2], "/o.mp4", duration=1, width=64, height=48),
        ffmpeg.probe_duration(user[0]),
        ffmpeg.probe_video_size(user[0]),
    ]
    for call in calls:
        # The fakes return little; only the argv matters here.
        with contextlib.suppress(Exception):
            await call
    opened = 0
    for args in commands:
        check_input_policy(args)
        for index, token in enumerate(args):
            if token == "-i" and args[index + 1] in user:
                opened += 1
                assert args[index - len(R):index] == R, args
    assert opened >= len(calls)


# --- the reproduced exploit, end to end ---------------------------------------------------


@NEED_FFMPEG
@pytest.mark.parametrize("name", ["clip.mp4", "clip.mov", "clip.mkv", "clip.webm",
                                  "song.m4a", "song.mp3", "song.wav"])
@pytest.mark.parametrize("reference", ["absolute", "relative", "file_url"])
async def test_disguised_playlists_are_never_followed(tmp_path: Path, name: str,
                                                      reference: str) -> None:
    # The referenced "other file" is a FIFO: opening it would block, so a fast,
    # clean failure proves FFmpeg never opened it.
    other = tmp_path / "other"
    other.mkdir()
    os.mkfifo(other / "secret.ts")
    target = {"absolute": str(other / "secret.ts"), "relative": "other/secret.ts",
              "file_url": f"file://{other / 'secret.ts'}"}[reference]
    disguised = tmp_path / name
    disguised.write_bytes(_playlist(target))
    media_type = MediaType.audio if name.startswith("song") else MediaType.video
    started = time.monotonic()
    with pytest.raises(MetadataProbeError):
        await asyncio.wait_for(probe_media(disguised, media_type), 10)
    with pytest.raises(ToolError):
        await asyncio.wait_for(
            ffmpeg.video_transcode(str(disguised), str(tmp_path / "out.mkv"), None, None), 10)
    with pytest.raises(waveform.WaveformGenerationError):
        await asyncio.wait_for(waveform.generate_waveform(disguised), 10)
    assert time.monotonic() - started < 8
    assert not (tmp_path / "out.mkv").exists()


@NEED_FFMPEG
async def test_disguised_concat_list_and_image_are_refused(tmp_path: Path) -> None:
    secret = _secret_media(tmp_path / "other" / "secret.ts")
    concat = tmp_path / "list.mp4"
    concat.write_text(f"ffconcat version 1.0\nfile {secret}\n", encoding="utf-8")
    with pytest.raises(MetadataProbeError):
        await probe_media(concat, MediaType.video)
    still = tmp_path / "photo.png"
    still.write_bytes(_playlist(str(secret)))
    with pytest.raises(ToolError):
        await ffmpeg.still_to_clip(str(still), str(tmp_path / "s.mp4"), duration=1,
                                   width=64, height=48)
    assert not (tmp_path / "s.mp4").exists()


@NEED_FFMPEG
async def test_shot_detection_checks_the_file_before_opencv(tmp_path: Path,
                                                            monkeypatch) -> None:
    secret = _secret_media(tmp_path / "other" / "secret.ts")
    disguised = tmp_path / "clip.mp4"
    disguised.write_bytes(_playlist(str(secret)))

    def opencv_must_not_run(*_args):
        raise AssertionError("OpenCV opened a file the restricted probe refused")

    monkeypatch.setattr(pyscenedetect, "_detect_sync", opencv_must_not_run)
    with pytest.raises(ToolError, match="could not be opened"):
        await pyscenedetect.detect_shots(str(disguised))
    monkeypatch.undo()
    real = tmp_path / "real.mp4"
    _ff("-f", "lavfi", "-i", "color=c=blue:s=64x48:d=1", "-pix_fmt", "yuv420p", str(real))
    assert isinstance(await pyscenedetect.detect_shots(str(real)), list)  # runs normally


@NEED_FFMPEG
async def test_agent_fused_render_refuses_a_disguised_source(tmp_path: Path) -> None:
    secret = _secret_media(tmp_path / "other" / "secret.ts")
    disguised = tmp_path / "clip.mp4"
    disguised.write_bytes(_playlist(str(secret)))
    vid = "44444444-4444-4444-4444-444444444444"
    plan = Plan.model_validate({"type": "plan", "steps": [
        {"id": "lower", "capability_id": "video.lowerthird", "asset": vid,
         "params": {"text": "Name"}},
    ]})
    group = compile_execution_plan(plan, {vid: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "f" * 64,
        "technical_metadata_status": "pending", "has_audio_stream": None,
    }})["nodes"][0]
    output = tmp_path / "out.mp4"
    with pytest.raises(Exception):  # noqa: B017 - any failure; the output is what matters
        await render_group(disguised, output, tmp_path, group)
    assert not output.exists() or output.stat().st_size == 0


async def _upload(client: AsyncClient, pid: str, name: str, body: bytes, mime: str) -> dict:
    response = await client.post(f"{API}/projects/{pid}/assets",
                                 files={"file": (name, io.BytesIO(body), mime)})
    assert response.status_code == 201, response.text
    return response.json()


@NEED_FFMPEG
async def test_disguised_upload_is_accepted_but_never_processed(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    secret = _secret_media(tmp_path / "other" / "secret.ts")
    pid = await make_project("disguised-upload")
    clip = await _upload(client, pid, "clip.mp4", _playlist(str(secret)), "video/mp4")
    # Upload is unchanged (201); the file is never probed as available media.
    assert clip["technical_metadata_status"] in ("pending", "unavailable")
    assert clip.get("has_video_stream") in (None, False)
    music = tmp_path / "m.wav"
    _ff("-f", "lavfi", "-i", "sine=d=1", str(music))
    bed = await _upload(client, pid, "m.wav", music.read_bytes(), "audio/wav")
    before = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    for body in (
        {"capability_id": "video.transcode", "asset_id": clip["id"],
         "input": {"format": "mkv"}},
        {"capability_id": "video.mute", "asset_id": clip["id"], "input": {}},
        {"capability_id": "video.audio.extract", "asset_id": clip["id"], "input": {}},
        {"capability_id": "video.thumbnail", "asset_id": clip["id"],
         "input": {"timestamp": 0}},
        {"capability_id": "video.shot.detect", "asset_id": clip["id"], "input": {}},
        {"capability_id": "audio.mix", "asset_id": clip["id"],
         "input": {"music_asset_id": bed["id"], "mode": "replace"}},
        {"capability_id": "video.compose",
         "input": {"asset_ids": [clip["id"]], "width": 64, "height": 48}},
    ):
        response = await client.post(f"{API}/jobs", json={"project_id": pid, **body})
        assert response.status_code == 201, (body["capability_id"], response.text)
        job = (await client.get(f"{API}/jobs/{response.json()['id']}")).json()
        assert job["status"] == "failed", (body["capability_id"], job)
        assert job["output"] is None
        assert str(tmp_path) not in (job["error"] or "") and "secret" not in job["error"]
    after = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    assert after == before


# --- every accepted format still works ---------------------------------------------------


def _fixture(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    video = ["-f", "lavfi", "-i", "testsrc=s=64x48:d=1", "-f", "lavfi", "-i", "sine=d=1",
             "-shortest"]
    audio = ["-f", "lavfi", "-i", "sine=d=1"]
    recipes = {
        "a.mp4": [*video, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"],
        "a.mov": [*video, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"],
        "a.m4v": [*video, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                  "-f", "mp4"],
        "a.mkv": [*video, "-c:v", "ffv1", "-c:a", "flac"],
        "a.webm": [*video, "-c:v", "libvpx-vp9", "-c:a", "libopus"],
        "a.avi": [*video, "-c:v", "mpeg4", "-c:a", "mp3"],
        "a.mp3": audio, "a.wav": audio, "a.flac": audio,
        "a.ogg": [*audio, "-c:a", "libvorbis"],
        "opus.ogg": [*audio, "-c:a", "libopus"],
        "a.m4a": [*audio, "-c:a", "aac", "-f", "ipod"],
    }
    if name in recipes:
        _ff(*recipes[name], str(path))
    else:
        subprocess.run(["convert", "-size", "32x24", "xc:red", str(path)], check=True,
                       timeout=30)
    return path


@NEED_FFMPEG
@pytest.mark.parametrize("name", ["a.mp4", "a.mov", "a.m4v", "a.mkv", "a.webm", "a.avi"])
async def test_every_video_format_probes_and_decodes(tmp_path: Path, name: str) -> None:
    source = _fixture(tmp_path, name)
    metadata = await probe_media(source, MediaType.video)
    assert metadata.status.value == "available" and metadata.has_video_stream
    await ffmpeg.video_transcode(str(source), str(tmp_path / "out.mkv"), None, None)
    assert (tmp_path / "out.mkv").stat().st_size > 0
    await ffmpeg.video_thumbnail(str(source), str(tmp_path / "t.png"), 0.1)
    assert (tmp_path / "t.png").stat().st_size > 0


@NEED_FFMPEG
@pytest.mark.parametrize("name", ["a.mp3", "a.wav", "a.flac", "a.ogg", "opus.ogg", "a.m4a"])
async def test_every_audio_format_probes_and_decodes(tmp_path: Path, name: str) -> None:
    source = _fixture(tmp_path, name)
    metadata = await probe_media(source, MediaType.audio)
    assert metadata.status.value == "available" and metadata.has_audio_stream
    assert (await waveform.generate_waveform(source)).bucket_count > 0
    await ffmpeg.audio_transcode(str(source), str(tmp_path / "out.wav"), None)
    assert (tmp_path / "out.wav").stat().st_size > 0


@NEED_FFMPEG
@pytest.mark.parametrize("name", ["a.png", "a.jpg", "a.jpeg", "a.webp", "a.tif", "a.tiff"])
async def test_every_image_format_becomes_a_clip(tmp_path: Path, name: str) -> None:
    source = _fixture(tmp_path, name)
    await ffmpeg.still_to_clip(str(source), str(tmp_path / "s.mp4"), duration=1,
                               width=64, height=48)
    assert (tmp_path / "s.mp4").stat().st_size > 0


def test_media_metadata_probes_use_the_allowlist() -> None:
    import inspect

    source = inspect.getsource(media_metadata)
    assert source.count("*UNTRUSTED_INPUT") >= 3


# --- Task #18D: effective values, fused argv, reuse invalidation --------------------------

W = FORMAT_WHITELIST


@pytest.mark.parametrize("args", [
    # The override found in review #18C: FFmpeg uses the last value.
    ["ffmpeg", "-y", *R, "-format_whitelist", "hls,mpegts", "-i", "/a.mp4", "/o.mkv"],
    ["ffmpeg", "-y", *R, "-protocol_whitelist", "file,http", "-i", "/a.mp4", "/o.mkv"],
    # Duplicates are refused even when the last values look right.
    ["ffmpeg", "-y", "-format_whitelist", "hls", *R, "-i", "/a.mp4", "/o.mkv"],
    ["ffmpeg", "-y", *R, *R, "-i", "/a.mp4", "/o.mkv"],
    # Malformed: an option without a value, an -i without a file.
    ["ffmpeg", "-y", "-protocol_whitelist", "file", "-format_whitelist", "-i", "/a.mp4"],
    ["ffmpeg", "-y", *R, "-i"],
    # A forced format is an exemption only alone and only for trusted formats.
    ["ffmpeg", "-y", "-f", "lavfi", *R, "-i", "/a.mp4", "/o.mkv"],
    ["ffmpeg", "-y", "-f", "lavfi", "-f", "hls", "-i", "/a.mp4", "/o.mkv"],
    ["ffmpeg", "-y", *R, "-f", "mov", "-i", "/a.mp4", "/o.mkv"],
    # ffprobe: overrides, no value, positional only, two inputs.
    ["ffprobe", "-v", "error", *R, "-format_whitelist", "hls", "-i", "/a.mp4"],
    ["ffprobe", "-v", "error", *R, "-i"],
    ["ffprobe", "-v", "error", *R, "/a.mp4"],
    ["ffprobe", "-v", "error", *R, "-i", "/a.mp4", "-i", "/b.mp4"],
    ["ffprobe", "-v", "error", "-f", "srt", *R, "-i", "/s.srt"],
])
def test_effective_policy_values_are_enforced(args: list[str]) -> None:
    with pytest.raises(UnrestrictedInputError):
        check_input_policy(args)


@pytest.mark.parametrize("args", [
    ["ffmpeg", "-y", "-format_whitelist", W, "-protocol_whitelist", "file", "-i", "/a.mp4",
     "/o.mkv"],                                                     # either order
    ["ffmpeg", "-y", *R, "-i", "/a.mp4", *R, "-i", "/b.wav", *R, "-i", "/c.png", "/o.mp4"],
    ["ffmpeg", "-y", "-loop", "1", "-t", "2", *R, "-i", "/c.png", "/o.mp4"],
    ["ffprobe", "-v", "error", "-of", "json", *R, "-i", "/a.mp4", "-show_streams"],
])
def test_legitimate_commands_still_pass(args: list[str]) -> None:
    check_input_policy(args)


@NEED_FFMPEG
async def test_the_review_override_is_refused_before_launch(tmp_path: Path,
                                                             monkeypatch) -> None:
    secret = _secret_media(tmp_path / "other" / "secret.ts")
    disguised = tmp_path / "clip.mp4"
    disguised.write_bytes(_playlist(str(secret)))
    out = tmp_path / "out.mkv"
    override = ["ffmpeg", "-v", "error", "-y", *R, "-format_whitelist", "hls,mpegts",
                "-i", str(disguised), "-c", "copy", str(out)]
    launched: list[tuple] = []
    real_spawn = process_policy.asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        launched.append(args)
        return await real_spawn(*args, **kwargs)

    monkeypatch.setattr(process_policy.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(ToolError, match="^Media processing failed.$"):
        await asyncio.wait_for(runner.run(override), 10)
    assert launched == [] and not out.exists()


async def test_fused_render_command_restricts_each_input(tmp_path: Path,
                                                         monkeypatch) -> None:
    """Reach render_group's final FFmpeg command (earlier probes are mocked)."""
    from app.agent import render
    from app.tools import imagemagick

    source = tmp_path / "source.mp4"
    source.write_bytes(b"not decoded: FFmpeg is mocked")
    overlays: list[str] = []
    commands: list[list[str]] = []

    async def no_check(*_args, **_kwargs) -> None:
        return None

    async def has_audio(_path: str) -> bool:
        return True

    async def draw_overlay(dst: str, *_args, **_kwargs) -> None:
        overlays.append(dst)
        Path(dst).write_bytes(b"png")

    async def capture(args: list[str], **_kwargs) -> str:
        commands.append(list(args))
        return ""

    monkeypatch.setattr(render, "_check_trim_windows", no_check)
    monkeypatch.setattr(render, "_check_crop_bounds", no_check)
    monkeypatch.setattr(render, "_check_rendered_video", no_check)
    monkeypatch.setattr(ffmpeg, "_has_audio", has_audio)
    monkeypatch.setattr(imagemagick, "render_lowerthird", draw_overlay)
    monkeypatch.setattr(runner, "run", capture)
    vid = "55555555-5555-5555-5555-555555555555"
    plan = Plan.model_validate({"type": "plan", "steps": [
        {"id": "trim", "capability_id": "video.trim", "asset": vid,
         "params": {"start": 0, "end": 1}},
        {"id": "lower", "capability_id": "video.lowerthird", "asset": "@trim",
         "params": {"text": "Name", "duration": 1}},
    ]})
    group = compile_execution_plan(plan, {vid: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "c" * 64,
        "technical_metadata_status": "available", "width": 320, "height": 180,
        "has_audio_stream": True, "duration_seconds": 5.0,
    }})["nodes"][0]
    assert group["kind"] == "render_group"
    await render_group(source, tmp_path / "out.mp4", tmp_path, group)
    assert len(commands) == 1 and len(overlays) == 1
    fused = commands[0]
    check_input_policy(fused)
    inputs = [fused[index + 1] for index, token in enumerate(fused) if token == "-i"]
    assert inputs == [str(source), overlays[0]]   # nothing else is opened
    for index, token in enumerate(fused):
        if token == "-i":
            assert fused[index - len(R):index] == R, fused


async def _reuse_fixture(client: AsyncClient, make_project, name: str):
    from app.core.database import async_session_factory
    from app.models.agent import AgentPlan, AgentPlanAttempt, AgentSession
    from app.models.asset import Asset
    from app.models.enums import AgentPlanStatus, TechnicalMetadataStatus

    pid = await make_project(name)
    ids = []
    for file_name, body in (("in.mp4", b"input-video"), ("old.mp4", b"old-output"),
                            ("new.mp4", b"new-output")):
        ids.append((await _upload(client, pid, file_name, body, "video/mp4"))["id"])
    async with async_session_factory() as session:
        for output in ids[1:]:
            asset = await session.get(Asset, __import__("uuid").UUID(output))
            asset.technical_metadata_status = TechnicalMetadataStatus.available.value
        agent_session = AgentSession(project_id=__import__("uuid").UUID(pid), asset_ids=[])
        session.add(agent_session)
        await session.flush()
        plan = AgentPlan(session_id=agent_session.id, status=AgentPlanStatus.failed,
                         steps=[], step_runs=[], result_asset_ids=[])
        session.add(plan)
        await session.flush()
        earlier = AgentPlanAttempt(plan_id=plan.id, attempt_number=1, status="failed",
                                   job_ids=[], result_asset_ids=[], step_runs=[])
        current = AgentPlanAttempt(plan_id=plan.id, attempt_number=2, status="running",
                                   job_ids=[], result_asset_ids=[], step_runs=[])
        session.add_all([earlier, current])
        await session.commit()
        return pid, ids, plan.id, earlier.id, current


async def test_reuse_from_before_input_confinement_is_not_selected(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    import uuid as uuid_module
    from datetime import UTC, datetime

    from app.agent import reuse
    from app.core.database import async_session_factory
    from app.models.enums import JobStatus
    from app.models.job import Job

    pid, (source, old_out, new_out), plan_id, earlier_id, current = await _reuse_fixture(
        client, make_project, "reuse-policy")
    project_id = uuid_module.UUID(pid)
    payloads: list[dict] = []
    real_hash = reuse._canonical_json_hash

    def capture(value):
        payloads.append(dict(value))
        return real_hash(value)

    monkeypatch.setattr(reuse, "_canonical_json_hash", capture)
    async with async_session_factory() as session:
        identity = await reuse.build_execution_identity(
            session, project_id=project_id, capability_id="video.mute",
            asset_id=uuid_module.UUID(source), params={})
        again = await reuse.build_execution_identity(
            session, project_id=project_id, capability_id="video.mute",
            asset_id=uuid_module.UUID(source), params={})
        group = {"identity_hash": "a" * 64, "source_checksum": "b" * 64}
        fused = reuse.render_group_identity(project_id, group, optimizer_version="v")
        fused_again = reuse.render_group_identity(project_id, group, optimizer_version="v")
    assert identity and identity == again and fused and fused == fused_again   # stable
    assert payloads[0]["media_input_policy"] == "ffmpeg-input-confinement-v1"
    # The identities as computed before this policy: the same payload without it.
    legacy, legacy_fused = (
        real_hash({k: v for k, v in payload.items() if k != "media_input_policy"})
        for payload in (payloads[0], payloads[2])
    )
    assert legacy != identity and legacy_fused != fused

    def job(identity_value: str, output: str, finished: datetime) -> Job:
        return Job(project_id=project_id, capability_id="video.mute", status=JobStatus.succeeded,
                   input={}, progress=100, finished_at=finished, agent_plan_id=plan_id,
                   agent_attempt_id=earlier_id, execution_identity=identity_value,
                   output={"outputs": [{"asset_id": output, "media_type": "video",
                                        "filename": "o.mp4"}]})

    now = datetime.now(UTC)
    async with async_session_factory() as session:
        session.add_all([job(legacy, old_out, now), job(legacy_fused, old_out, now)])
        await session.commit()
        for value in (identity, fused):
            assert await reuse.find_reusable_output(
                session, plan_id=plan_id, current_attempt=current, project_id=project_id,
                identity=value) is None
        session.add_all([job(identity, new_out, now), job(fused, new_out, now)])
        await session.commit()
        for value in (identity, fused):
            found = await reuse.find_reusable_output(
                session, plan_id=plan_id, current_attempt=current, project_id=project_id,
                identity=value)
            assert found is not None and str(found.asset.id) == new_out
        # Project scoping is unchanged: another project never sees these Jobs.
        assert await reuse.find_reusable_output(
            session, plan_id=plan_id, current_attempt=current,
            project_id=uuid_module.uuid4(), identity=identity) is None
