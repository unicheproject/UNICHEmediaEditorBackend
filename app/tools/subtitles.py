"""Content validation for SRT/WebVTT subtitle files, before FFmpeg reads them.

FFmpeg picks a demuxer from the file contents. Binary data named .srt can be
detected as MPEG-TS and make the libass `subtitles` filter spin until the
runner timeout; other content yields "successful" output without subtitles.
So subtitle files are checked here in Python first (bounded size, strict
UTF-8, SRT/WebVTT structure, at least one renderable cue), and only then
read by FFprobe with the detected demuxer forced, to confirm that FFmpeg sees
the cues. Nothing is rewritten: the original file is what FFmpeg renders.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable, Iterator
from enum import StrEnum
from itertools import chain
from pathlib import Path
from typing import NamedTuple

from app.core.errors import ToolError, ValidationError
from app.tools import runner

MAX_SUBTITLE_BYTES = 10 * 1024 * 1024
# Only for the subtitle check below; rendering keeps the runner's default.
PROBE_TIMEOUT_SECONDS = 30.0


class SubtitleFormat(StrEnum):
    """Values are FFmpeg demuxer names."""

    srt = "srt"
    webvtt = "webvtt"


# C0 controls other than tab/LF/CR, DEL and C1 controls never occur in text
# subtitles; their presence means binary or mis-decoded data.
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_ASS_SECTION = re.compile(r"\[(script info|v4\+? styles|events)\]", re.IGNORECASE)
_SRT_INDEX = re.compile(r"\d{1,9}")
# Same shape the FFmpeg srt demuxer reads: H:M:S with ',' or '.' before the
# milliseconds; anything after the end time (e.g. X1:.. coordinates) is ignored.
_SRT_TIMING = re.compile(
    r"(\d{1,3}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})[ \t]*-->[ \t]*"
    r"(\d{1,3}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})(?:[ \t].*)?"
)
# A whole line that is an SRT timing pair with a broken separator (`->`, an
# en dash from autocorrect, a missing arrow...). FFmpeg reads such a cue as
# text of the previous one. Only a complete pair, optionally followed by SRT
# coordinates, counts: two times followed by words stay ordinary text.
_SRT_TS = r"\d{1,3}:\d{1,2}:\d{1,2}[,.]\d{1,3}"
_SRT_BROKEN_TIMING = re.compile(
    rf"{_SRT_TS}(?:[ \t]*[^\w\s]{{1,4}}[ \t]*|[ \t]+){_SRT_TS}"
    r"(?:[ \t]+X1:\d+[ \t]+X2:\d+[ \t]+Y1:\d+[ \t]+Y2:\d+)?"
)
# WebVTT: optional hours, mm:ss.fff, an arrow with optional spaces (FFmpeg
# reads `-->` without them correctly), then optional cue settings.
_VTT_TIMING = re.compile(
    r"(?:(\d{1,3}):)?(\d{2}):(\d{2})\.(\d{1,3})[ \t]*-->[ \t]*"
    r"(?:(\d{1,3}):)?(\d{2}):(\d{2})\.(\d{1,3})(?:[ \t].*)?"
)
# Blocks the FFmpeg 5.1 webvtt demuxer cannot handle: it reports success but
# drops the cues that follow (measured). They are rejected, never stripped.
_VTT_UNSUPPORTED_BLOCKS = ("STYLE", "REGION")


def _millis(hours: str | None, minutes: str, seconds: str, fraction: str) -> int | None:
    h, m, s, ms = int(hours or 0), int(minutes), int(seconds), int(fraction)
    if m > 59 or s > 59:
        return None
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def _timing(match: re.Match[str]) -> tuple[int, int] | None:
    start = _millis(*match.group(1, 2, 3, 4))
    end = _millis(*match.group(5, 6, 7, 8))
    return None if start is None or end is None else (start, end)


def _lines(text: str) -> Iterator[str]:
    """Lines of LF-normalized text, one at a time (no list of all lines)."""
    start = 0
    while (end := text.find("\n", start)) >= 0:
        yield text[start:end]
        start = end + 1
    yield text[start:]


class _Block(NamedTuple):
    """A blank-line separated block, summarized: only what validation needs."""

    first: str
    second: str | None
    lines: int
    has_arrow: bool


def _blocks(text: str) -> Iterator[_Block]:
    """Yield blocks lazily; whitespace-only lines separate blocks.

    Memory stays proportional to the input: a block keeps its first two lines
    and a count, however long it is.
    """
    first: str | None = None
    second: str | None = None
    count = 0
    arrow = False
    for line in _lines(text):
        if line and not line.isspace():  # same as line.strip(), no copy
            if count == 0:
                first = line
            elif count == 1:
                second = line
            count += 1
            arrow = arrow or "-->" in line
        elif first is not None:
            yield _Block(first, second, count, arrow)
            first, second, count, arrow = None, None, 0, False
    if first is not None:
        yield _Block(first, second, count, arrow)


class _Tally:
    """Counts cues in constant space; a cue is checked once it is complete."""

    def __init__(self) -> None:
        self.cues = 0
        self.renderable = 0
        self._open: tuple[int, int] | None = None
        self._has_text = False

    def add(self, start: int, end: int, *, has_text: bool) -> None:
        self._close()
        self.cues += 1
        self._open, self._has_text = (start, end), has_text

    def add_text(self) -> None:
        self._has_text = True

    def _close(self) -> None:
        if self._open is None:
            return
        start, end = self._open
        if end < start:
            raise ValidationError(f"Subtitle cue {self.cues} ends before it starts")
        if not self._has_text:
            raise ValidationError(f"Subtitle cue {self.cues} has no text")
        if end > start:
            self.renderable += 1
        self._open = None

    def finish(self) -> int:
        self._close()
        if self.renderable == 0:
            raise ValidationError("Subtitle file contains no renderable cues")
        return self.renderable


def _broken_timing(block: _Block) -> bool:
    """The block's timing position holds a timing pair without a valid `-->`.

    The timing position is the first line, or the second after a cue number.
    Only those (at most two) lines are inspected, so parsing stays linear.
    """
    first = block.first.strip()
    if block.second is not None and _SRT_INDEX.fullmatch(first):
        candidate = block.second.strip()
    else:
        candidate = first
    return (
        candidate[:1].isdigit()
        and "-->" not in candidate
        and _SRT_BROKEN_TIMING.fullmatch(candidate) is not None
    )


def _count_srt(blocks: Iterable[_Block]) -> int:
    tally = _Tally()
    for block in blocks:
        if _broken_timing(block):
            raise ValidationError(
                f"Subtitle cue {tally.cues + 1} has an invalid timing line; "
                "the arrow must be -->"
            )
        if not block.has_arrow and tally.cues:
            tally.add_text()  # no timing line anywhere: text joined to the cue
            continue
        has_index = bool(_SRT_INDEX.fullmatch(block.first.strip()))
        timing_line = block.second if has_index and block.second is not None else block.first
        match = _SRT_TIMING.fullmatch(timing_line.strip())
        if match is None:
            if "-->" in timing_line:
                raise ValidationError(f"Subtitle cue {tally.cues + 1} has an invalid timestamp")
            if tally.cues == 0:
                raise ValidationError("Subtitle file is not a valid SRT or WebVTT file")
            # A blank line inside a cue: FFmpeg joins the text to that cue.
            tally.add_text()
            continue
        times = _timing(match)
        if times is None:
            raise ValidationError(f"Subtitle cue {tally.cues + 1} has an invalid timestamp")
        # Only the first cue needs its number (FFmpeg's format detection reads
        # it); later unnumbered cues are read normally.
        tally.add(*times, has_text=block.lines > (2 if has_index else 1))
    return tally.finish()


def _count_vtt(header: _Block, blocks: Iterable[_Block]) -> int:
    if header.has_arrow:
        # FFmpeg skips the whole header block, so such a cue would be lost.
        raise ValidationError(
            "WebVTT needs a blank line between the WEBVTT header and the first cue"
        )
    tally = _Tally()
    for block in blocks:
        keyword = block.first.split(maxsplit=1)[0]
        if keyword == "NOTE":
            continue
        if keyword in _VTT_UNSUPPORTED_BLOCKS:
            raise ValidationError(f"WebVTT {keyword} blocks are not supported")
        if "-->" in block.first:
            timing_line, text_lines = block.first, block.lines - 1
        elif block.second is not None and "-->" in block.second:
            timing_line, text_lines = block.second, block.lines - 2
        else:
            # Not a cue, NOTE or header: FFmpeg may drop the next cue (measured).
            raise ValidationError(
                f"WebVTT block after cue {tally.cues} is not a cue or NOTE"
            )
        match = _VTT_TIMING.fullmatch(timing_line.strip())
        times = _timing(match) if match else None
        if match is None or times is None:
            raise ValidationError(f"Subtitle cue {tally.cues + 1} has an invalid timestamp")
        if len(match.group(4)) != 3 or len(match.group(8)) != 3:
            # FFmpeg 5.1 reads ".5" as 5 ms, not 500 ms (measured).
            raise ValidationError(
                f"Subtitle cue {tally.cues + 1} needs three digits after the decimal "
                "point (for example 00:00:01.500)"
            )
        tally.add(*times, has_text=text_lines > 0)
    return tally.finish()


def check_subtitle_bytes(data: bytes) -> tuple[SubtitleFormat, int]:
    """Classify SRT/WebVTT by content; return the format and renderable cue count."""
    if not data.strip():
        raise ValidationError("Subtitle file is empty")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValidationError(
            "Subtitle file is not UTF-8 text; save it as UTF-8 and upload it again"
        ) from None
    text = text.removeprefix("\ufeff")
    if _CONTROL.search(text):
        raise ValidationError("Subtitle file contains binary data")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = _blocks(text)
    header = next(blocks, None)
    if header is None:
        raise ValidationError("Subtitle file is empty")
    first = header.first
    if first.startswith("WEBVTT") and first[6:7] in ("", " ", "\t"):
        return SubtitleFormat.webvtt, _count_vtt(header, blocks)
    if _ASS_SECTION.fullmatch(first.strip()):
        raise ValidationError("ASS/SSA subtitles are not supported; use SRT or WebVTT")
    if _SRT_INDEX.fullmatch(first.strip()) and _broken_timing(header):
        raise ValidationError(
            "Subtitle cue 1 has an invalid timing line; the arrow must be -->"
        )
    if not (
        _SRT_INDEX.fullmatch(first.strip())
        and header.second is not None
        and "-->" in header.second
    ):
        raise ValidationError("Subtitle file is not a valid SRT or WebVTT file")
    return SubtitleFormat.srt, _count_srt(chain([header], blocks))


def _read_bounded(path: Path) -> bytes:
    try:
        size = path.stat().st_size
        if size > MAX_SUBTITLE_BYTES:
            raise ValidationError("Subtitle file is larger than the 10 MiB limit")
        with path.open("rb") as handle:
            data = handle.read(MAX_SUBTITLE_BYTES + 1)
    except OSError:
        raise ValidationError("Subtitle file could not be read") from None
    if len(data) > MAX_SUBTITLE_BYTES:  # grew after stat
        raise ValidationError("Subtitle file is larger than the 10 MiB limit")
    return data


def _probed_packets(output: str) -> int:
    try:
        streams = json.loads(output).get("streams")
        stream = streams[0] if isinstance(streams, list) and len(streams) == 1 else None
        if not isinstance(stream, dict) or stream.get("codec_type") != "subtitle":
            return 0
        packets = stream.get("nb_read_packets")
        return int(packets) if isinstance(packets, (str, int)) else 0
    except (ValueError, TypeError, AttributeError):
        return 0


async def validate_subtitle_file(path: str | Path) -> SubtitleFormat:
    """Reject unusable subtitle files before any FFmpeg render reads them."""
    file = Path(path).absolute()
    data = await asyncio.to_thread(_read_bounded, file)
    subtitle_format, _cues = await asyncio.to_thread(check_subtitle_bytes, data)
    try:
        output = await runner.run_stdout(
            [
                "ffprobe", "-v", "error", "-f", subtitle_format.value, "-i", str(file),
                "-select_streams", "s", "-count_packets",
                "-show_entries", "stream=codec_type,nb_read_packets", "-of", "json",
            ],
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except ToolError as exc:
        if exc.message == "Media processing tool unavailable.":
            raise
        if exc.message == "Media processing timed out.":
            raise ValidationError("Subtitle file check timed out") from None
        raise ValidationError("Subtitle file could not be read as subtitles") from None
    if _probed_packets(output) < 1:
        raise ValidationError("Subtitle file contains no cues the renderer can read")
    return subtitle_format
