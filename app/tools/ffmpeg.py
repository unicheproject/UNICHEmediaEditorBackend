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


async def probe_video_size(path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream via ffprobe."""
    out = await runner.run_stdout(
        [
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", path,
        ],
        timeout=60,
    )
    try:
        stream = json.loads(out)["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
        return 0, 0


async def _has_audio(path: str) -> bool:
    out = await runner.run_stdout(
        [
            FFPROBE, "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", path,
        ],
        timeout=60,
    )
    return bool(out.strip())


async def _concat_normalize(src: str, dst: str, width: int, height: int) -> None:
    """Re-encode a clip to uniform video + audio params for safe concatenation.

    The clip's own audio is preserved; a silent track is synthesised only when
    the source has none, so heterogeneous inputs (different sizes / with or
    without sound) still concat with matching stream parameters.
    """
    has_audio = await _has_audio(src)
    args = [*_BASE, "-i", src]
    if not has_audio:
        args += _ANULL
    audio_in = "0:a:0" if has_audio else "1:a:0"
    args += [
        "-vf", _scale_pad(width, height),
        "-map", "0:v:0", "-map", audio_in,
        "-ar", "44100", "-ac", "2",
        *_PIX, "-c:v", "libx264", "-c:a", "aac", "-shortest", dst,
    ]
    await runner.run(args)


async def video_concat(sources: list[str], dst: str, work_dir: Path) -> None:
    """Concatenate video clips, normalizing each to the first clip's resolution.

    The ffmpeg `concat` filter/demuxer require identical stream parameters
    across inputs, so each clip is first scaled/padded to a common size and
    re-encoded to uniform codecs before a stream-copy concat.
    """
    width, height = await probe_video_size(sources[0])
    if width <= 0 or height <= 0:
        width, height = 1920, 1080

    normalized: list[Path] = []
    for i, src in enumerate(sources):
        out = work_dir / f"_concat_norm_{i:03d}.mp4"
        await _concat_normalize(src, str(out), width, height)
        normalized.append(out)

    listing = work_dir / "_concat_list.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in normalized), encoding="utf-8")
    await runner.run(
        [*_BASE, "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", dst]
    )


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


# --- composition ---------------------------------------------------------
#
# Composition clips carry a uniform silent stereo audio track and yuv420p video,
# so heterogeneous segments (title cards, slideshows, plain clips) concatenate
# cleanly. `compose` lays an optional single audio bed over the whole timeline.

FPS = 30
_SAR = "setsar=1"
_PIX = ["-pix_fmt", "yuv420p"]
_ANULL = ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]


def _scale_pad(width: int, height: int) -> str:
    w, h = _num(width, "width"), _num(height, "height")
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,{_SAR},fps={FPS}"
    )


async def still_to_clip(image: str, dst: str, duration: float, width: int, height: int) -> None:
    """A still image -> a video clip of `duration` with a silent audio track."""
    await runner.run(
        [
            *_BASE,
            "-loop", "1", "-t", _num(duration, "duration"), "-i", image,
            *_ANULL,
            "-vf", _scale_pad(width, height),
            *_PIX, "-c:v", "libx264", "-c:a", "aac", "-shortest", dst,
        ]
    )


async def slideshow(
    images: list[str], dst: str, seconds_per_image: float, width: int, height: int
) -> None:
    """Concatenate still images into one video clip (+ silent audio)."""
    args = [*_BASE]
    for img in images:
        args += ["-loop", "1", "-t", _num(seconds_per_image, "seconds_per_image"), "-i", img]
    args += _ANULL
    n = len(images)
    sp = _scale_pad(width, height)
    chains = "".join(f"[{i}:v]{sp}[v{i}];" for i in range(n))
    concat_in = "".join(f"[v{i}]" for i in range(n))
    filt = f"{chains}{concat_in}concat=n={n}:v=1:a=0[v]"
    args += [
        "-filter_complex", filt,
        "-map", "[v]", "-map", f"{n}:a",
        *_PIX, "-c:v", "libx264", "-c:a", "aac", "-shortest", dst,
    ]
    await runner.run(args)


def _escape_subs(path: str) -> str:
    # The subtitles filter needs ':' and '\' escaped inside the filter string.
    return path.replace("\\", "\\\\").replace(":", "\\:")


async def subtitle_embed(video: str, subtitle: str, dst: str, mode: str = "soft") -> None:
    if mode == "burn":
        await runner.run(
            [*_BASE, "-i", video, "-vf", f"subtitles='{_escape_subs(subtitle)}'", *_PIX, dst]
        )
    else:  # soft mux
        await runner.run(
            [*_BASE, "-i", video, "-i", subtitle, "-c", "copy", "-c:s", "mov_text", dst]
        )


async def audio_mix(
    media: str, music: str, dst: str, music_volume: float = 0.3, mode: str = "mix"
) -> None:
    """Lay `music` under `media`'s audio. mode: mix | duck (≈low mix) | replace."""
    vol = music_volume if mode != "duck" else min(music_volume, 0.2)
    if mode == "replace":
        await runner.run(
            [*_BASE, "-i", media, "-i", music,
             "-map", "0:v?", "-map", "1:a", "-c:v", "copy", "-shortest", dst]
        )
        return
    filt = f"[1:a]volume={vol}[m];[0:a][m]amix=inputs=2:duration=first[a]"
    await runner.run(
        [*_BASE, "-i", media, "-i", music,
         "-filter_complex", filt, "-map", "0:v?", "-map", "[a]",
         "-c:v", "copy", dst]
    )


async def _normalize_segment(src: str, dst: str, width: int, height: int) -> None:
    """Re-encode a segment to uniform params (silent audio) for safe concat."""
    await runner.run(
        [
            *_BASE, "-i", src, *_ANULL,
            "-map", "0:v", "-map", "1:a",
            "-vf", _scale_pad(width, height),
            *_PIX, "-c:v", "libx264", "-c:a", "aac", "-shortest", dst,
        ]
    )


async def compose(
    segments: list[str],
    dst: str,
    *,
    width: int,
    height: int,
    work_dir: Path,
    audio_path: str | None = None,
    audio_mode: str = "mix",
    music_volume: float = 0.3,
    subtitle_path: str | None = None,
) -> None:
    """Timeline render: normalize -> concat -> optional audio bed -> optional burn."""
    # 1. normalize each segment to identical codec/params
    normalized: list[Path] = []
    for i, seg in enumerate(segments):
        out = work_dir / f"_norm_{i:03d}.mp4"
        await _normalize_segment(seg, str(out), width, height)
        normalized.append(out)

    # 2. concat via the demuxer (copy; params already identical)
    listing = work_dir / "_concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in normalized), encoding="utf-8")
    concat = work_dir / "_concat.mp4"
    await runner.run(
        [*_BASE, "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(concat)]
    )

    current = concat
    # 3. optional audio bed over the whole timeline
    if audio_path:
        with_audio = work_dir / "_audio.mp4"
        await audio_mix(str(current), audio_path, str(with_audio),
                        music_volume=music_volume, mode=audio_mode)
        current = with_audio

    # 4. optional subtitle burn-in (final re-encode)
    if subtitle_path:
        burned = work_dir / "_subs.mp4"
        await subtitle_embed(str(current), subtitle_path, str(burned), mode="burn")
        current = burned

    # 5. move final result to dst
    if str(current) != dst:
        await runner.run([*_BASE, "-i", str(current), "-c", "copy", dst])
