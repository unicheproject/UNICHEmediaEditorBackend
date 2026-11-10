"""Task #17C: SRT/WebVTT content is validated before FFmpeg reads a subtitle file."""

from __future__ import annotations

import io
import random
import re
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.agent import reuse
from app.capabilities.handlers.subtitle_autogenerate import parse_segments, to_srt
from app.core.errors import ToolError, ValidationError
from app.tools import ffmpeg, subtitles
from app.tools.subtitles import (
    MAX_SUBTITLE_BYTES,
    SubtitleFormat,
    check_subtitle_bytes,
    validate_subtitle_file,
)
from tests.test_composition_clip_timing import NEED_FFMPEG
from tests.test_subtitle_burn_style import _count, _frame, _text_pixels, _video

API = "/api/v1"
CUE = "1\n00:00:00,200 --> 00:00:01,800\nHello\n"
VTT_CUE = "00:00:00.200 --> 00:00:01.800\nHello\n"


def _random(seed: int, size: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(size))


# FFmpeg 5.1 detects these 300 random bytes as MPEG-TS (score 2); the libass
# `subtitles` filter then spins until killed (measured in Task #17B). This file
# must never reach FFmpeg: every test below rejects it in Python first.
MPEGTS_LIKE = _random(0, 300)

REJECTED: dict[str, tuple[bytes, str]] = {
    "empty": (b"", "empty"),
    "whitespace": (b"  \n\n\t\r\n", "empty"),
    "vtt header only": (b"WEBVTT\n\n", "no renderable cues"),
    "vtt header and note only": (b"WEBVTT\n\nNOTE nothing here\n", "no renderable cues"),
    "random binary": (_random(17, 4096), "UTF-8|binary"),
    "mpeg-ts lookalike": (MPEGTS_LIKE, "UTF-8|binary"),
    "nul bytes": (b"\x00" * 64, "binary"),
    "png": (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "UTF-8|binary"),
    "plain text": (b"this is not a subtitle\nat all\n", "not a valid SRT or WebVTT"),
    "index only": (b"1\n\n", "not a valid SRT or WebVTT"),
    "srt without first number": (
        b"00:00:00,200 --> 00:00:01,500\nNo number\n", "not a valid SRT or WebVTT"
    ),
    "ass": (
        b"[Script Info]\nScriptType: v4.00+\n\n[Events]\nFormat: Layer, Start, End, "
        b"Style, Text\nDialogue: 0,0:00:00.20,0:00:01.50,Default,Hi\n",
        "ASS/SSA",
    ),
    "vtt style": (
        ("WEBVTT\n\nSTYLE\n::cue { color: red }\n\n" + VTT_CUE).encode(), "STYLE"
    ),
    "vtt style after cues": (
        ("WEBVTT\n\n" + VTT_CUE + "\nSTYLE\n::cue { color: red }\n").encode(), "STYLE"
    ),
    "vtt region": (("WEBVTT\n\nREGION\nid:fred\n\n" + VTT_CUE).encode(), "REGION"),
    "invalid timestamp": (b"1\n00:00:xx,200 --> 00:00:01,500\nBad\n", "invalid timestamp"),
    "minutes out of range": (b"1\n00:61:00,000 --> 00:62:00,000\nBad\n", "invalid timestamp"),
    "vtt one-digit fraction": (b"WEBVTT\n\n00:00.2 --> 00:01.500\nBad\n", "three digits"),
    "reversed": (b"1\n00:00:01,500 --> 00:00:00,200\nBack\n", "ends before it starts"),
    "missing text": (
        b"1\n00:00:00,200 --> 00:00:01,500\n\n2\n00:00:02,000 --> 00:00:03,000\nB\n",
        "cue 1 has no text",
    ),
    "utf-16": ("1\n00:00:00,200 --> 00:00:01,500\nΚαλημέρα\n".encode("utf-16"), "UTF-8"),
    "cp1253": ("1\n00:00:00,200 --> 00:00:01,500\nΚαλημέρα\n".encode("cp1253"), "UTF-8"),
    # Task #17C-FIX: WebVTT blocks that are not a cue or NOTE are rejected,
    # wherever they are (FFmpeg dropped the next cue after a "->" block).
    "vtt text block before cues": (
        ("WEBVTT\n\nrandom words\n\n" + VTT_CUE).encode(), "not a cue or NOTE"
    ),
    "vtt broken arrow between cues": (
        b"WEBVTT\n\n00:00.000 --> 00:01.000\nOne\n\n00:01.000 -> 00:02.000\nBroken\n\n"
        b"00:02.000 --> 00:03.000\nThree\n",
        "block after cue 1 is not a cue or NOTE",
    ),
    "vtt text block after cues": (
        ("WEBVTT\n\n" + VTT_CUE + "\ntrailing words\n").encode(), "not a cue or NOTE"
    ),
    "vtt identifier without timing": (
        b"WEBVTT\n\nintro\nno timing here\n", "not a cue or NOTE"
    ),
    "vtt malformed note": (("WEBVTT\n\nNOTE:x\n\n" + VTT_CUE).encode(), "not a cue or NOTE"),
    "vtt cue inside the header": (
        b"WEBVTT\n00:00.000 --> 00:01.000\nLost\n\n00:01.000 --> 00:02.000\nKept\n",
        "blank line between the WEBVTT header",
    ),
    "vtt two-digit fraction": (b"WEBVTT\n\n00:00.50 --> 00:01.500\nBad\n", "three digits"),
    "vtt minutes out of range": (b"WEBVTT\n\n00:61.000 --> 00:62.000\nBad\n",
                                 "invalid timestamp"),
    "vtt seconds out of range": (b"WEBVTT\n\n00:00:60.000 --> 00:01:01.000\nBad\n",
                                 "invalid timestamp"),
    "vtt reversed": (b"WEBVTT\n\n00:02.000 --> 00:01.000\nBack\n", "ends before it starts"),
    "vtt missing text": (b"WEBVTT\n\n00:00.000 --> 00:01.000\n", "cue 1 has no text"),
    # Task #17C-FIX2: an SRT timing pair with a broken arrow sits where a timing
    # line belongs; FFmpeg would show it as text of the previous cue.
    "srt ascii arrow between cues": (
        b"1\n00:00:00,000 --> 00:00:01,000\nOne\n\n2\n00:00:01,000 -> 00:00:02,000\nTwo\n\n"
        b"3\n00:00:02,000 --> 00:00:03,000\nThree\n",
        "cue 2 has an invalid timing line",
    ),
    "srt en dash arrow between cues": (
        "1\n00:00:00,000 --> 00:00:01,000\nOne\n\n2\n00:00:01,000 \u2013> 00:00:02,000\n"
        "Two\n\n3\n00:00:02,000 --> 00:00:03,000\nThree\n".encode(),
        "cue 2 has an invalid timing line",
    ),
    "srt broken timing opening a block": (
        "1\n00:00:00,000 --> 00:00:01,000\nOne\n\n00:00:01,000 \u2014 00:00:02,000\nTwo\n"
        .encode(),
        "cue 2 has an invalid timing line",
    ),
    "srt broken timing after the last cue": (
        b"1\n00:00:00,000 --> 00:00:01,000\nOne\n\n2\n00:00:01,000 => 00:00:02,000\nTwo\n",
        "cue 2 has an invalid timing line",
    ),
    "srt missing arrow": (
        b"1\n00:00:00,000 --> 00:00:01,000\nOne\n\n2\n00:00:01,000 00:00:02,000\nTwo\n",
        "cue 2 has an invalid timing line",
    ),
    "srt broken first cue": (
        b"1\n00:00:00,000 -> 00:00:01,000\nOne\n", "cue 1 has an invalid timing line"
    ),
    "zero-duration only": (
        b"1\n00:00:01,000 --> 00:00:01,000\nZero\n", "no renderable cues"
    ),
}

ACCEPTED: dict[str, tuple[bytes, SubtitleFormat, int]] = {
    "srt": (CUE.encode(), SubtitleFormat.srt, 1),
    "vtt": (("WEBVTT\n\n" + VTT_CUE).encode(), SubtitleFormat.webvtt, 1),
    "greek multiline": (
        "1\n00:00:00,200 --> 00:00:01,800\nΚαλημέρα κόσμε\nδεύτερη γραμμή\n".encode(),
        SubtitleFormat.srt, 1,
    ),
    "utf-8 bom": (("\ufeff" + CUE).encode(), SubtitleFormat.srt, 1),
    "crlf": (CUE.replace("\n", "\r\n").encode(), SubtitleFormat.srt, 1),
    "html tags": (
        b"1\n00:00:00,200 --> 00:00:01,800\n<i>italic</i> <b>bold</b>\n",
        SubtitleFormat.srt, 1,
    ),
    "dot milliseconds": (b"1\n00:00:00.200 --> 00:00:01.800\nDot\n", SubtitleFormat.srt, 1),
    "one-digit hour": (b"1\n0:00:00,200 --> 0:00:01,800\nOne\n", SubtitleFormat.srt, 1),
    "coordinates": (
        b"1\n00:00:00,200 --> 00:00:01,800 X1:10 X2:100 Y1:10 Y2:50\nAt\n",
        SubtitleFormat.srt, 1,
    ),
    "no final newline": (b"1\n00:00:00,200 --> 00:00:01,800\nEnd", SubtitleFormat.srt, 1),
    "out of order": (
        b"1\n00:00:01,000 --> 00:00:01,500\nB\n\n2\n00:00:00,200 --> 00:00:00,800\nA\n",
        SubtitleFormat.srt, 2,
    ),
    "outside any short video": (
        b"1\n01:00:10,000 --> 01:00:11,000\nLate\n", SubtitleFormat.srt, 1
    ),
    "zero-duration plus valid": (
        b"1\n00:00:01,000 --> 00:00:01,000\nZero\n\n2\n00:00:00,200 --> 00:00:01,800\nOk\n",
        SubtitleFormat.srt, 1,
    ),
    "blank line inside a cue": (
        b"1\n00:00:00,200 --> 00:00:01,000\nA\n\nstill A\n\n2\n00:00:01,000 --> "
        b"00:00:01,800\nB\n",
        SubtitleFormat.srt, 2,
    ),
    "later cue without number": (
        b"1\n00:00:00,200 --> 00:00:01,000\nA\n\n00:00:01,000 --> 00:00:01,800\nB\n",
        SubtitleFormat.srt, 2,
    ),
    "vtt short timestamps": (
        b"WEBVTT\n\n00:00.200 --> 00:01.800\nShort\n", SubtitleFormat.webvtt, 1
    ),
    "vtt id settings voice note": (
        "WEBVTT - title\n\nNOTE a comment\nspanning lines\n\nintro\n00:00:00.200 --> "
        "00:00:01.800 align:start line:0\n<v Narrator>Γεια σου</v>\nκαι πάλι\n".encode(),
        SubtitleFormat.webvtt, 1,
    ),
    "vtt content named srt": (("WEBVTT\n\n" + VTT_CUE).encode(), SubtitleFormat.webvtt, 1),
    "vtt arrow without spaces": (
        b"WEBVTT\n\n00:00:00.200-->00:00:01.800\nTight\n", SubtitleFormat.webvtt, 1
    ),
    "vtt arrow with one space": (
        b"WEBVTT\n\n00:00.200 -->00:01.800\nLeft\n\n00:02.000--> 00:03.000\nRight\n",
        SubtitleFormat.webvtt, 2,
    ),
    "vtt region setting and notes": (
        b"WEBVTT\n\nNOTE before\n\n00:00.200 --> 00:01.800 region:fred align:left\nA\n\n"
        b"NOTE between\n\nid-2\n00:02.000 --> 00:03.000\nB\n\nNOTE after\n",
        SubtitleFormat.webvtt, 2,
    ),
    "vtt header metadata": (
        b"WEBVTT\nKind: captions\nLanguage: el\n\n00:00.200 --> 00:01.800\nA\n",
        SubtitleFormat.webvtt, 1,
    ),
    "srt numbers and times in text": (
        "1\n00:00:00,200 --> 00:00:01,800\nΣτις 10:30\n\n2\n\n00:00:05,000 είναι η αρχή\n\n"
        "00:00:01,000 \u2013 00:00:02,000 was the window\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nΑπό 00:00:01,000 \u2013 00:00:02,000\n".encode(),
        SubtitleFormat.srt, 2,
    ),
    "srt time range inside cue text": (
        b"1\n00:00:00,200 --> 00:00:01,800\nSchedule:\n00:00:01,000 - 00:00:02,000\n",
        SubtitleFormat.srt, 1,
    ),
    "srt one-digit milliseconds": (
        b"1\n00:00:00,5 --> 00:00:01,5\nRaw ms\n", SubtitleFormat.srt, 1
    ),
    "unusual valid text": (
        "1\n00:00:00,200 --> 00:00:01,800\n♪ «Καλή» — 100% → <font color=\"#ff0\">ok</font>"
        " 😀 \u200b\u202bשלום\u202c\n".encode(),
        SubtitleFormat.srt, 1,
    ),
}


# --- the Python checks (no binaries) --------------------------------------------------


@pytest.mark.parametrize("case", sorted(REJECTED))
def test_invalid_content_is_rejected(case: str) -> None:
    data, message = REJECTED[case]
    with pytest.raises(ValidationError, match=message):
        check_subtitle_bytes(data)


@pytest.mark.parametrize("case", sorted(ACCEPTED))
def test_valid_content_is_accepted(case: str) -> None:
    data, expected_format, renderable = ACCEPTED[case]
    assert check_subtitle_bytes(data) == (expected_format, renderable)


def _parse_seconds(data: bytes) -> float:
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        check_subtitle_bytes(data)
        best = min(best, time.perf_counter() - started)
    return best


_FRAGMENT_HEAD = b"1\n00:00:00,000 --> 00:00:01,000\nA\n\n"


@pytest.mark.parametrize("fragment", [b"x\n\n", b"x\n", b"\n"],
                         ids=["continuation blocks", "one long cue", "blank lines"])
def test_fragmented_input_scales_linearly(fragment: bytes) -> None:
    """4x the input must cost about 4x, not 16x (Task #17C review: += was quadratic)."""
    small = _parse_seconds(_FRAGMENT_HEAD + fragment * 50_000)
    large = _parse_seconds(_FRAGMENT_HEAD + fragment * 200_000)
    assert large < 8 * max(small, 0.005), (small, large)


def test_fragmented_input_memory_stays_proportional() -> None:
    import tracemalloc

    data = _FRAGMENT_HEAD + b"x\n\n" * 50_000
    tracemalloc.start()
    try:
        assert check_subtitle_bytes(data) == (SubtitleFormat.srt, 1)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # Decoded text and its LF-normalized copy; no per-line or per-block lists.
    assert peak < 4 * len(data), (peak, len(data))


def test_autogenerated_srt_passes_validation() -> None:
    cues = parse_segments([
        {"start": 0.0001, "end": 0.0004, "text": "rounds to zero length"},
        {"start": 0.5, "end": 1.25, "text": "Καλημέρα, κόσμε"},
        {"start": 3725.5, "end": 3727.0, "text": "after an hour"},
    ])
    assert check_subtitle_bytes(to_srt(cues).encode("utf-8")) == (SubtitleFormat.srt, 2)


@pytest.mark.parametrize("case", ["random binary", "mpeg-ts lookalike", "png", "utf-16",
                                  "ass", "vtt style", "zero-duration only",
                                  "srt en dash arrow between cues",
                                  "srt broken timing opening a block"])
async def test_rejected_files_never_reach_ffprobe(monkeypatch, tmp_path: Path,
                                                  case: str) -> None:
    calls: list[list[str]] = []

    async def record(args: list[str], **_kwargs) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(subtitles.runner, "run_stdout", record)
    monkeypatch.setattr(subtitles.runner, "run", record)
    path = tmp_path / "subs.srt"
    path.write_bytes(REJECTED[case][0])
    with pytest.raises(ValidationError):
        await validate_subtitle_file(path)
    assert calls == []


async def test_size_limit_is_checked_before_reading(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "huge.srt"
    with path.open("wb") as handle:
        handle.truncate(MAX_SUBTITLE_BYTES + 1)  # sparse: nothing is written

    def no_read(*_args, **_kwargs):
        raise AssertionError("an oversized subtitle file must not be opened")

    monkeypatch.setattr(Path, "open", no_read)
    with pytest.raises(ValidationError, match="10 MiB"):
        await validate_subtitle_file(path)


async def test_missing_file_is_a_validation_error(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="could not be read") as raised:
        await validate_subtitle_file(tmp_path / "gone.srt")
    assert str(tmp_path) not in raised.value.message


@pytest.mark.parametrize(("output", "accepted"), [
    ('{"streams":[{"codec_type":"subtitle","nb_read_packets":"2"}]}', True),
    ('{"streams":[{"codec_type":"subtitle","nb_read_packets":"N/A"}]}', False),
    ('{"streams":[{"codec_type":"subtitle","nb_read_packets":"0"}]}', False),
    ('{"streams":[{"codec_type":"video","nb_read_packets":"3"}]}', False),
    ('{"streams":[]}', False),
    ("not json", False),
    ("", False),
])
async def test_probe_must_confirm_readable_cues(monkeypatch, tmp_path: Path,
                                               output: str, accepted: bool) -> None:
    seen: list[list[str]] = []

    async def probe(args: list[str], *, timeout: float) -> str:
        seen.append(args)
        assert timeout == subtitles.PROBE_TIMEOUT_SECONDS == 30.0
        return output

    monkeypatch.setattr(subtitles.runner, "run_stdout", probe)
    path = tmp_path / "subs.vtt"
    path.write_bytes(CUE.encode())  # SRT content: the demuxer follows content
    if accepted:
        assert await validate_subtitle_file(path) is SubtitleFormat.srt
    else:
        with pytest.raises(ValidationError, match="no cues the renderer can read"):
            await validate_subtitle_file(path)
    assert seen[0][:6] == ["ffprobe", "-v", "error", "-f", "srt", "-i"]


@pytest.mark.parametrize(("failure", "outcome"), [
    ("Media processing timed out.", "check timed out"),
    ("Media processing failed.", "could not be read as subtitles"),
])
async def test_probe_failures_become_validation_errors(monkeypatch, tmp_path: Path,
                                                      failure: str, outcome: str) -> None:
    async def probe(_args: list[str], **_kwargs) -> str:
        raise ToolError(failure)

    monkeypatch.setattr(subtitles.runner, "run_stdout", probe)
    path = tmp_path / "subs.srt"
    path.write_bytes(CUE.encode())
    with pytest.raises(ValidationError, match=outcome):
        await validate_subtitle_file(path)


async def test_missing_ffprobe_stays_a_tool_error(monkeypatch, tmp_path: Path) -> None:
    async def probe(_args: list[str], **_kwargs) -> str:
        raise ToolError("Media processing tool unavailable.")

    monkeypatch.setattr(subtitles.runner, "run_stdout", probe)
    path = tmp_path / "subs.srt"
    path.write_bytes(CUE.encode())
    with pytest.raises(ToolError, match="unavailable"):
        await validate_subtitle_file(path)


async def test_compose_rejects_subtitles_before_any_rendering(monkeypatch,
                                                              tmp_path: Path) -> None:
    calls: list[list[str]] = []

    async def record(args: list[str], **_kwargs) -> str:
        calls.append(args)
        return '{"streams":[{"duration":"1"}]}'

    monkeypatch.setattr(ffmpeg.runner, "run", record)
    monkeypatch.setattr(ffmpeg.runner, "run_stdout", record)
    subs = tmp_path / "subs.srt"
    subs.write_bytes(MPEGTS_LIKE)
    with pytest.raises(ValidationError):
        await ffmpeg.compose(["a.mp4", "b.mp4"], str(tmp_path / "out.mp4"), width=96,
                             height=64, work_dir=tmp_path, subtitle_path=str(subs))
    assert calls == []


async def test_compose_validates_subtitles_once(monkeypatch, tmp_path: Path) -> None:
    probes: list[list[str]] = []

    async def run(_args: list[str], **_kwargs) -> str:
        return ""

    async def run_stdout(args: list[str], **_kwargs) -> str:
        if "-count_packets" in args:
            probes.append(args)
            return '{"streams":[{"codec_type":"subtitle","nb_read_packets":"1"}]}'
        return '{"streams":[{"duration":"1"}]}'

    monkeypatch.setattr(ffmpeg.runner, "run", run)
    monkeypatch.setattr(ffmpeg.runner, "run_stdout", run_stdout)
    subs = tmp_path / "subs.srt"
    subs.write_bytes(CUE.encode())
    await ffmpeg.compose(["a.mp4"], str(tmp_path / "out.mp4"), width=96, height=64,
                         work_dir=tmp_path, subtitle_path=str(subs))
    assert len(probes) == 1


async def test_reuse_identity_changes_with_version_3(client: AsyncClient,
                                                     make_project, monkeypatch) -> None:
    from app.core.database import async_session_factory

    assert reuse.REUSE_VERSIONS["video.subtitle.embed"] == "3"
    assert reuse.REUSE_VERSIONS["audio.mix"] == "2"
    project_id = uuid.UUID(await make_project("subtitle reuse v3"))
    ids = []
    for name, body, mime in (("v.mp4", b"video", "video/mp4"),
                             ("s.srt", CUE.encode(), "application/x-subrip")):
        response = await client.post(f"{API}/projects/{project_id}/assets",
                                     files={"file": (name, body, mime)})
        assert response.status_code == 201, response.text
        ids.append(response.json()["id"])
    params = {"subtitle_asset_id": ids[1], "mode": "burn"}
    async with async_session_factory() as session:
        current = await reuse.build_execution_identity(
            session, project_id=project_id, capability_id="video.subtitle.embed",
            asset_id=uuid.UUID(ids[0]), params=params,
        )
        with monkeypatch.context() as patch:
            patch.setitem(reuse.REUSE_VERSIONS, "video.subtitle.embed", "2")
            previous = await reuse.build_execution_identity(
                session, project_id=project_id, capability_id="video.subtitle.embed",
                asset_id=uuid.UUID(ids[0]), params=params,
            )
    assert current is not None and previous is not None and current != previous


# --- real FFprobe / FFmpeg --------------------------------------------------------------


@NEED_FFMPEG
@pytest.mark.parametrize("case", sorted(ACCEPTED))
async def test_ffprobe_confirms_every_accepted_case(tmp_path: Path, case: str) -> None:
    data, expected_format, _renderable = ACCEPTED[case]
    path = tmp_path / ("subs.vtt" if expected_format is SubtitleFormat.srt else "subs.srt")
    path.write_bytes(data)  # deliberately the "wrong" extension: content decides
    assert await validate_subtitle_file(path) is expected_format


def _subtitle_text(video: Path) -> str:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:s:0", "-f", "srt", "-"],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout


@NEED_FFMPEG
async def test_unstyled_output_matches_the_previous_commands(tmp_path: Path) -> None:
    video = _video(tmp_path / "v.mp4")
    subs = tmp_path / "s.srt"
    subs.write_text("1\n00:00:00,000 --> 00:00:02,000\nΚαλημέρα HELLO\n", encoding="utf-8")
    old_burn, new_burn = tmp_path / "old_burn.mp4", tmp_path / "new_burn.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
                    f"subtitles='{subs}'", "-pix_fmt", "yuv420p", str(old_burn)],
                   check=True, timeout=30)
    await ffmpeg.subtitle_embed(str(video), str(subs), str(new_burn), mode="burn")
    assert _frame(new_burn) == _frame(old_burn)
    assert len(_text_pixels(_frame(new_burn))) > 50

    old_soft, new_soft = tmp_path / "old_soft.mp4", tmp_path / "new_soft.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(subs),
                    "-c", "copy", "-c:s", "mov_text", str(old_soft)], check=True, timeout=30)
    await ffmpeg.subtitle_embed(str(video), str(subs), str(new_soft), mode="soft")
    assert _subtitle_text(new_soft) == _subtitle_text(old_soft)
    assert "Καλημέρα HELLO" in _subtitle_text(new_soft)


@NEED_FFMPEG
async def test_relaxed_vtt_arrows_keep_their_timing(tmp_path: Path) -> None:
    video = _video(tmp_path / "v.mp4")
    subs = tmp_path / "s.vtt"
    subs.write_text("WEBVTT\n\n00:00.200 -->00:00.800\nLeft\n\n00:01.000--> 00:01.400\n"
                    "Right\n\n00:00:01.600-->00:00:01.900\nTight\n", encoding="utf-8")
    assert await validate_subtitle_file(subs) is SubtitleFormat.webvtt
    soft = tmp_path / "soft.mp4"
    await ffmpeg.subtitle_embed(str(video), str(subs), str(soft), mode="soft")
    track = _subtitle_text(soft)
    for timing, text in (("00:00:00,200 --> 00:00:00,800", "Left"),
                         ("00:00:01,000 --> 00:00:01,400", "Right"),
                         ("00:00:01,600 --> 00:00:01,900", "Tight")):
        assert f"{timing}\n{text}" in track, track
    burned = tmp_path / "burn.mp4"
    await ffmpeg.subtitle_embed(str(video), str(subs), str(burned), mode="burn")
    for seconds, visible in ((0.5, True), (0.9, False), (1.2, True), (1.5, False),
                             (1.7, True)):
        assert (len(_text_pixels(_frame(burned, seconds))) > 20) is visible, seconds


async def _upload(client: AsyncClient, pid: str, name: str, body: bytes, mime: str) -> str:
    response = await client.post(f"{API}/projects/{pid}/assets",
                                 files={"file": (name, io.BytesIO(body), mime)})
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def _job(client: AsyncClient, body: dict) -> dict:
    response = await client.post(f"{API}/jobs", json=body)
    assert response.status_code == 201, response.text
    return (await client.get(f"{API}/jobs/{response.json()['id']}")).json()


async def _download(client: AsyncClient, job: dict, path: Path) -> Path:
    path.write_bytes((await client.get(job["output"]["outputs"][0]["download_path"])).content)
    return path


@NEED_FFMPEG
@pytest.mark.parametrize(("name", "body", "extra"), [
    ("subs.srt", "1\n00:00:00,000 --> 00:00:02,000\nΚαλημέρα κόσμε\n", {"mode": "soft"}),
    ("subs.vtt", "WEBVTT\n\n00:00.000 --> 00:02.000\nΚαλημέρα κόσμε\n", {"mode": "soft"}),
    ("subs.srt", "1\n00:00:00,000 --> 00:00:02,000\nΚαλημέρα κόσμε\n", {"mode": "burn"}),
    ("subs.vtt", "WEBVTT\n\nNOTE hi\n\n00:00.000 --> 00:02.000\nΚαλημέρα κόσμε\n",
     {"mode": "burn", "text_colour": "#FF0000", "font_size": 40, "outline": 0}),
])
async def test_valid_subtitle_jobs_succeed(client: AsyncClient, make_project,
                                           tmp_path: Path, name: str, body: str,
                                           extra: dict) -> None:
    pid = await make_project(f"subs-ok-{name}-{sorted(extra.items())}")
    video = await _upload(client, pid, "clip.mp4",
                          _video(tmp_path / "v.mp4").read_bytes(), "video/mp4")
    subs = await _upload(client, pid, name, body.encode(), "text/plain")
    job = await _job(client, {
        "capability_id": "video.subtitle.embed", "project_id": pid, "asset_id": video,
        "input": {"subtitle_asset_id": subs, **extra},
    })
    assert job["status"] == "succeeded", job.get("error")
    out = await _download(client, job, tmp_path / "out.mp4")
    if extra["mode"] == "soft":
        assert "Καλημέρα κόσμε" in _subtitle_text(out)
    elif "text_colour" in extra:
        assert _count(_frame(out), lambda r, g, b: r - max(g, b) > 80) > 100
    else:
        assert len(_text_pixels(_frame(out))) > 50


@NEED_FFMPEG
@pytest.mark.parametrize(("case", "name", "mode", "message"), [
    ("mpeg-ts lookalike", "subs.srt", "burn", "UTF-8|binary"),
    ("random binary", "subs.vtt", "soft", "UTF-8|binary"),
    ("png", "subs.srt", "soft", "UTF-8|binary"),
    ("vtt header only", "subs.vtt", "burn", "no renderable cues"),
    ("vtt style", "subs.vtt", "soft", "STYLE"),
    ("ass", "subs.srt", "burn", "ASS/SSA"),
    ("reversed", "subs.srt", "burn", "ends before it starts"),
    ("vtt broken arrow between cues", "subs.vtt", "soft", "not a cue or NOTE"),
    ("srt en dash arrow between cues", "subs.srt", "burn", "invalid timing line"),
])
async def test_invalid_subtitle_jobs_fail_without_an_asset(
    client: AsyncClient, make_project, tmp_path: Path, case: str, name: str, mode: str,
    message: str,
) -> None:
    pid = await make_project(f"subs-bad-{case}")
    video = await _upload(client, pid, "clip.mp4",
                          _video(tmp_path / "v.mp4").read_bytes(), "video/mp4")
    subs = await _upload(client, pid, name, REJECTED[case][0], "text/plain")
    before = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    started = time.monotonic()
    job = await _job(client, {
        "capability_id": "video.subtitle.embed", "project_id": pid, "asset_id": video,
        "input": {"subtitle_asset_id": subs, "mode": mode},
    })
    assert time.monotonic() - started < 5
    assert job["status"] == "failed" and job["output"] is None
    assert job["error"] and re.search(message, job["error"])
    assert str(tmp_path) not in job["error"] and "/tmp" not in job["error"]
    after = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    assert after == before


@NEED_FFMPEG
@pytest.mark.parametrize("valid", [True, False])
async def test_direct_compose_with_subtitles(client: AsyncClient, make_project,
                                             tmp_path: Path, valid: bool) -> None:
    pid = await make_project(f"compose-subs-{valid}")
    video = await _upload(client, pid, "clip.mp4",
                          _video(tmp_path / "v.mp4").read_bytes(), "video/mp4")
    body = (b"1\n00:00:00,000 --> 00:00:02,000\nHELLO\n" if valid else MPEGTS_LIKE)
    subs = await _upload(client, pid, "subs.srt", body, "text/plain")
    before = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    job = await _job(client, {
        "capability_id": "video.compose", "project_id": pid,
        "input": {"asset_ids": [video], "subtitle_asset_id": subs,
                  "width": 320, "height": 180},
    })
    if valid:
        assert job["status"] == "succeeded", job.get("error")
        out = await _download(client, job, tmp_path / "composed.mp4")
        assert len(_text_pixels(_frame(out))) > 50
    else:
        assert job["status"] == "failed" and job["output"] is None
        after = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
        assert after == before
