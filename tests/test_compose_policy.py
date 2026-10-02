"""Tests for typed default and lossless composition render policies."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from app.capabilities import registry
from app.tools import ffmpeg
from app.tools.compose_policy import (
    DEFAULT_COMPOSE_RENDER_POLICY,
    LOSSLESS_COMPOSE_RENDER_POLICY,
    ComposeAudioCodec,
    ComposeContainer,
    ComposePixelFormat,
    ComposeRenderPolicy,
    ComposeVideoCodec,
)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_FFPROBE = shutil.which("ffprobe") is not None
NEED_FFMPEG = pytest.mark.skipif(
    not (HAVE_FFMPEG and HAVE_FFPROBE), reason="needs ffmpeg and ffprobe"
)

_LOSSY_ENCODERS = {
    "libx264",
    "libx265",
    "libvpx",
    "libvpx-vp9",
    "aac",
    "libmp3lame",
    "libopus",
    "libvorbis",
}


def _run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    return proc.stdout


def _make_lossless_video(path: str, color: str, duration: float = 0.4) -> None:
    _run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c={color}:s=96x64:r=10:d={duration}",
            "-an", "-c:v", "ffv1", "-pix_fmt", "yuv444p16le", path,
        ]
    )


def _make_audio_bed(path: str) -> None:
    _run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
            "-i", "sine=frequency=440:sample_rate=44100:duration=2",
            "-c:a", "pcm_s16le", path,
        ]
    )


def _probe(path: str) -> dict:
    return json.loads(
        _run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=format_name,duration:stream=codec_type,codec_name,width,height,pix_fmt,sample_fmt",
                "-of", "json", path,
            ]
        )
    )


def _frame_hashes(path: str) -> list[str]:
    output = _run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-f", "framemd5", "-"]
    )
    return [
        line.split(",")[-1].strip()
        for line in output.splitlines()
        if line and not line.startswith("#")
    ]


def _encoder_arguments(command: list[str]) -> set[str]:
    encoders: set[str] = set()
    for index, argument in enumerate(command[:-1]):
        if argument in {"-c", "-codec", "-c:v", "-c:a", "-codec:v", "-codec:a"}:
            encoders.add(command[index + 1])
    return encoders


def test_policy_definitions_are_typed_and_match_renderer_contract() -> None:
    assert isinstance(DEFAULT_COMPOSE_RENDER_POLICY, ComposeRenderPolicy)
    assert isinstance(LOSSLESS_COMPOSE_RENDER_POLICY, ComposeRenderPolicy)
    assert DEFAULT_COMPOSE_RENDER_POLICY.container == ComposeContainer.MP4
    assert DEFAULT_COMPOSE_RENDER_POLICY.suffix == ".mp4"
    assert DEFAULT_COMPOSE_RENDER_POLICY.video_codec == ComposeVideoCodec.H264
    assert DEFAULT_COMPOSE_RENDER_POLICY.audio_codec == ComposeAudioCodec.AAC
    assert DEFAULT_COMPOSE_RENDER_POLICY.pixel_format == ComposePixelFormat.YUV420P
    assert LOSSLESS_COMPOSE_RENDER_POLICY.container == ComposeContainer.MATROSKA
    assert LOSSLESS_COMPOSE_RENDER_POLICY.suffix == ".mkv"
    assert LOSSLESS_COMPOSE_RENDER_POLICY.video_codec == ComposeVideoCodec.FFV1
    assert LOSSLESS_COMPOSE_RENDER_POLICY.audio_codec == ComposeAudioCodec.FLAC
    assert LOSSLESS_COMPOSE_RENDER_POLICY.pixel_format == ComposePixelFormat.YUV444P16LE


def test_video_compose_public_schema_does_not_expose_render_policy() -> None:
    schema = registry.get("video.compose").input_schema
    assert set(schema["properties"]) == {
        "asset_ids",
        "width",
        "height",
        "audio_asset_id",
        "audio_mode",
        "music_volume",
        "subtitle_asset_id",
    }
    assert schema["required"] == ["asset_ids"]


@pytest.mark.parametrize("mode", ["mix", "duck", "replace"])
async def test_default_policy_preserves_audio_bed_stream_copy_and_mp4_encoders(
    monkeypatch, tmp_path, mode: str
) -> None:
    commands: list[list[str]] = []

    async def fake_run(args: list[str], *, timeout: float = 600.0) -> str:
        commands.append(args)
        return ""

    async def fake_run_stdout(args: list[str], *, timeout: float = 600.0) -> str:
        return "0\n"

    monkeypatch.setattr(ffmpeg.runner, "run", fake_run)
    monkeypatch.setattr(ffmpeg.runner, "run_stdout", fake_run_stdout)
    await ffmpeg.compose(
        ["segment.mp4"],
        str(tmp_path / "out.mp4"),
        width=96,
        height=64,
        work_dir=tmp_path,
        audio_path="bed.wav",
        audio_mode=mode,
    )

    normalization = commands[0]
    assert normalization[-1].endswith("_norm_000.mp4")
    assert normalization[normalization.index("-c:v") + 1] == "libx264"
    assert normalization[normalization.index("-c:a") + 1] == "aac"
    mix_command = next(command for command in commands if "bed.wav" in command)
    assert mix_command[mix_command.index("-c:v") + 1] == "copy"
    assert mix_command[mix_command.index("-c:a") + 1] == "aac"
    assert any(command[-1].endswith("_concat.mp4") for command in commands)
    assert any(command[-1].endswith("_audio.mp4") for command in commands)


@pytest.mark.parametrize("mode", [None, "mix", "duck", "replace"])
@pytest.mark.parametrize("subtitle", [False, True])
async def test_lossless_invocations_never_select_a_lossy_encoder(
    monkeypatch, tmp_path, mode: str | None, subtitle: bool
) -> None:
    commands: list[list[str]] = []

    async def fake_run(args: list[str], *, timeout: float = 600.0) -> str:
        commands.append(args)
        return ""

    async def fake_run_stdout(args: list[str], *, timeout: float = 600.0) -> str:
        return "0\n"

    monkeypatch.setattr(ffmpeg.runner, "run", fake_run)
    monkeypatch.setattr(ffmpeg.runner, "run_stdout", fake_run_stdout)
    bed = "bed.wav" if mode is not None else None
    subtitle_path = "captions.srt" if subtitle else None
    suffix = LOSSLESS_COMPOSE_RENDER_POLICY.suffix

    await ffmpeg.compose(
        ["first.mkv", "second.mkv"],
        str(tmp_path / f"out{suffix}"),
        width=96,
        height=64,
        work_dir=tmp_path,
        audio_path=bed,
        audio_mode=mode or "mix",
        subtitle_path=subtitle_path,
        policy=LOSSLESS_COMPOSE_RENDER_POLICY,
    )

    encoders = set().union(*(_encoder_arguments(command) for command in commands))
    assert "ffv1" in encoders
    assert "flac" in encoders
    assert not any(
        encoder in _LOSSY_ENCODERS or encoder.startswith("libvpx")
        for encoder in encoders
    )
    assert all(".mp4" not in argument for command in commands for argument in command)
    assert any(command[-1].endswith("_concat.mkv") for command in commands)
    if mode is not None:
        mix_command = next(
            command
            for command in commands
            if "-filter_complex" in command
            or ("-shortest" in command and "bed.wav" in command)
        )
        assert "-c:v" in mix_command and mix_command[mix_command.index("-c:v") + 1] == "copy"
        assert "-c:a" in mix_command and mix_command[mix_command.index("-c:a") + 1] == "flac"
    if subtitle:
        subtitle_command = next(
            command
            for command in commands
            if any("subtitles=" in argument for argument in command)
        )
        assert "-c:v" in subtitle_command
        assert subtitle_command[subtitle_command.index("-c:v") + 1] == "ffv1"
        assert "-c:a" in subtitle_command
        assert subtitle_command[subtitle_command.index("-c:a") + 1] == "flac"


@NEED_FFMPEG
async def test_lossless_compose_is_ffv1_flac_and_concat_preserves_normalized_frames(
    tmp_path, monkeypatch
) -> None:
    red, blue = tmp_path / "red.mkv", tmp_path / "blue.mkv"
    _make_lossless_video(str(red), "red")
    _make_lossless_video(str(blue), "blue")
    expected_red, expected_blue = tmp_path / "expected-red.mkv", tmp_path / "expected-blue.mkv"
    await ffmpeg._normalize_segment(
        str(red), str(expected_red), 96, 64, LOSSLESS_COMPOSE_RENDER_POLICY
    )
    await ffmpeg._normalize_segment(
        str(blue), str(expected_blue), 96, 64, LOSSLESS_COMPOSE_RENDER_POLICY
    )

    commands: list[list[str]] = []
    actual_run = ffmpeg.runner.run

    async def capture(args: list[str], *, timeout: float = 600.0) -> str:
        commands.append(list(args))
        return await actual_run(args, timeout=timeout)

    monkeypatch.setattr(ffmpeg.runner, "run", capture)
    output = tmp_path / "joined.mkv"
    await ffmpeg.compose(
        [str(red), str(blue), str(red)],
        str(output),
        width=96,
        height=64,
        work_dir=tmp_path,
        policy=LOSSLESS_COMPOSE_RENDER_POLICY,
    )

    metadata = _probe(str(output))
    streams = {stream["codec_type"]: stream for stream in metadata["streams"]}
    assert "matroska" in metadata["format"]["format_name"]
    assert streams["video"]["codec_name"] == "ffv1"
    assert streams["audio"]["codec_name"] == "flac"
    assert (streams["video"]["width"], streams["video"]["height"]) == (96, 64)
    assert streams["video"]["pix_fmt"] == "yuv444p16le"
    assert streams["audio"]["sample_fmt"] == "s32"
    expected_duration = 3 * float(_probe(str(expected_red))["format"]["duration"])
    assert abs(float(metadata["format"]["duration"]) - expected_duration) <= 0.02
    assert _frame_hashes(str(output)) == (
        _frame_hashes(str(expected_red))
        + _frame_hashes(str(expected_blue))
        + _frame_hashes(str(expected_red))
    )

    encoders = set().union(*(_encoder_arguments(command) for command in commands))
    assert encoders == {"ffv1", "flac", "copy"}
    assert not any(
        encoder in _LOSSY_ENCODERS or encoder.startswith("libvpx")
        for encoder in encoders
    )


@NEED_FFMPEG
@pytest.mark.parametrize("mode", ["mix", "duck", "replace"])
async def test_lossless_audio_bed_modes_keep_ffv1_and_use_flac(
    tmp_path, mode: str
) -> None:
    source, bed = tmp_path / "source.mkv", tmp_path / "bed.wav"
    _make_lossless_video(str(source), "blue")
    _make_audio_bed(str(bed))
    output = tmp_path / f"{mode}.mkv"

    await ffmpeg.compose(
        [str(source)],
        str(output),
        width=96,
        height=64,
        work_dir=tmp_path,
        audio_path=str(bed),
        audio_mode=mode,
        policy=LOSSLESS_COMPOSE_RENDER_POLICY,
    )

    streams = {stream["codec_type"]: stream for stream in _probe(str(output))["streams"]}
    assert streams["video"]["codec_name"] == "ffv1"
    assert streams["audio"]["codec_name"] == "flac"
