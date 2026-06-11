"""FFmpeg wrappers for deterministic video and audio operations.

Each function builds an argument list and delegates to runner.run. Output
container/codec is generally chosen by FFmpeg from the output file extension.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.errors import ToolError
from app.tools import runner

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# Common prefix: overwrite output, only log real errors.
_BASE = [FFMPEG, "-y", "-loglevel", "error"]


def _num(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"Parameter '{name}' must be a number, got {value!r}")
    return str(value)


async def probe_duration(path: str) -> float:
    """Return media duration in seconds via ffprobe."""
    out = await runner.run_stdout(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ],
        timeout=60,
    )
    try:
        return float(json.loads(out)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return 0.0


# --- video ---------------------------------------------------------------


async def video_trim(src: str, dst: str, start: float, end: float) -> None:
    await runner.run(
        [*_BASE, "-i", src, "-ss", _num(start, "start"), "-to", _num(end, "end"), dst]
    )


async def video_split(src: str, dst_pattern: str, markers: list[float]) -> None:
    csv = ",".join(_num(m, "markers") for m in markers)
    await runner.run(
        [
            *_BASE,
            "-i",
            src,
            "-f",
            "segment",
            "-segment_times",
            csv,
            "-reset_timestamps",
            "1",
            "-c",
            "copy",
            dst_pattern,
        ]
    )


async def video_concat(sources: list[str], dst: str) -> None:
    args = [*_BASE]
    for s in sources:
        args += ["-i", s]
    n = len(sources)
    streams = "".join(f"[{i}:v][{i}:a]" for i in range(n))
    filt = f"{streams}concat=n={n}:v=1:a=1[v][a]"
    args += ["-filter_complex", filt, "-map", "[v]", "-map", "[a]", dst]
    await runner.run(args)


async def video_transcode(
    src: str, dst: str, video_codec: str | None, audio_codec: str | None
) -> None:
    args = [*_BASE, "-i", src]
    if video_codec:
        args += ["-c:v", video_codec]
    if audio_codec:
        args += ["-c:a", audio_codec]
    args.append(dst)
    await runner.run(args)


async def video_mute(src: str, dst: str) -> None:
    await runner.run([*_BASE, "-i", src, "-an", "-c:v", "copy", dst])


async def video_crop(src: str, dst: str, x: int, y: int, width: int, height: int) -> None:
    crop = f"crop={_num(width, 'width')}:{_num(height, 'height')}:{_num(x, 'x')}:{_num(y, 'y')}"
    await runner.run([*_BASE, "-i", src, "-vf", crop, dst])


async def video_resize(src: str, dst: str, width: int, height: int) -> None:
    scale = f"scale={_num(width, 'width')}:{_num(height, 'height')}"
    await runner.run([*_BASE, "-i", src, "-vf", scale, dst])


async def video_thumbnail(src: str, dst: str, timestamp: float) -> None:
    await runner.run(
        [*_BASE, "-ss", _num(timestamp, "timestamp"), "-i", src, "-frames:v", "1", dst]
    )


# --- audio ---------------------------------------------------------------


async def audio_trim(src: str, dst: str, start: float, end: float) -> None:
    await runner.run(
        [*_BASE, "-i", src, "-ss", _num(start, "start"), "-to", _num(end, "end"), dst]
    )


async def audio_concat(sources: list[str], dst: str) -> None:
    args = [*_BASE]
    for s in sources:
        args += ["-i", s]
    n = len(sources)
    streams = "".join(f"[{i}:a]" for i in range(n))
    filt = f"{streams}concat=n={n}:v=0:a=1[a]"
    args += ["-filter_complex", filt, "-map", "[a]", dst]
    await runner.run(args)


async def audio_gain(src: str, dst: str, gain_db: float) -> None:
    await runner.run(
        [*_BASE, "-i", src, "-filter:a", f"volume={_num(gain_db, 'gain_db')}dB", dst]
    )


async def audio_normalize(src: str, dst: str, target_i: float) -> None:
    filt = f"loudnorm=I={_num(target_i, 'target_i')}:TP=-1.5:LRA=11"
    await runner.run([*_BASE, "-i", src, "-filter:a", filt, dst])


async def audio_fade(src: str, dst: str, fade_in: float, fade_out: float) -> None:
    filters: list[str] = []
    if fade_in:
        filters.append(f"afade=t=in:st=0:d={_num(fade_in, 'fade_in')}")
    if fade_out:
        duration = await probe_duration(src)
        start = max(duration - float(fade_out), 0.0)
        filters.append(f"afade=t=out:st={start}:d={_num(fade_out, 'fade_out')}")
    if not filters:
        raise ToolError("audio.fade requires fade_in and/or fade_out > 0")
    await runner.run([*_BASE, "-i", src, "-filter:a", ",".join(filters), dst])


async def audio_transcode(src: str, dst: str, codec: str | None) -> None:
    args = [*_BASE, "-i", src]
    if codec:
        args += ["-c:a", codec]
    args.append(dst)
    await runner.run(args)


def segment_outputs(dst_pattern: str) -> list[Path]:
    """Glob the files produced by video_split (e.g. seg%03d.mp4 -> seg000.mp4...)."""
    p = Path(dst_pattern)
    # Translate a printf-style pattern's directory + a glob of the stem.
    stem = p.name.split("%")[0]
    return sorted(p.parent.glob(f"{stem}*{p.suffix}"))
