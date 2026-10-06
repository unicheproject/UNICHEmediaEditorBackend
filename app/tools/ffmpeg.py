"""FFmpeg wrappers for deterministic video and audio operations.

Each function builds an argument list and delegates to runner.run. Output
container/codec is generally chosen by FFmpeg from the output file extension.
"""

from __future__ import annotations

import json
import math
from contextlib import suppress
from pathlib import Path

from app.capabilities.media_options import VIDEO_TRANSCODE_OPTIONS
from app.core.errors import ToolError, ValidationError
from app.models.enums import MediaType
from app.services.media_metadata import MetadataProbeError, probe_media, probe_video_timeline
from app.tools import runner
from app.tools.compose_policy import (
    DEFAULT_COMPOSE_RENDER_POLICY,
    ComposeRenderPolicy,
)
from app.tools.export_policy import (
    CompositionPolicyId,
    EffectiveExportSettings,
    ExportContainer,
    FitPolicy,
)

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


async def video_split(src: str, dst_pattern: str, markers: list[float]) -> list[Path]:
    """Render and validate all frame-owned intervals before Asset publication."""
    if not isinstance(markers, list) or not markers:
        raise ValidationError("Split markers must be a non-empty list of seconds")
    for marker in markers:
        if isinstance(marker, bool) or not isinstance(marker, (int, float)):
            raise ValidationError("Split markers must be finite numbers")
        try:
            finite = math.isfinite(marker)
        except OverflowError:
            finite = False
        if not finite:
            raise ValidationError("Split markers must be finite numbers")
        if marker <= 0:
            raise ValidationError("Split marker must be greater than 0 seconds")
    if any(right <= left for left, right in zip(markers, markers[1:], strict=False)):
        raise ValidationError("Split markers must be strictly increasing")
    options = VIDEO_TRANSCODE_OPTIONS.get(Path(dst_pattern).suffix.lstrip(".").lower())
    if options is None:
        raise ValidationError("Video split does not support this output container")
    try:
        metadata = await probe_media(Path(src), MediaType.video)
        if not metadata.has_video_stream or not metadata.width or not metadata.height:
            raise MetadataProbeError("no_video_stream")
        timeline = await probe_video_timeline(
            Path(src), tuple(markers), frame_rate=metadata.frame_rate
        )
    except MetadataProbeError as exc:
        messages = {
            "marker_outside_video": "Split marker must be before the end of the video",
            "empty_video_interval": "Split markers must produce distinct non-empty video intervals",
        }
        raise ValidationError(messages.get(
            exc.reason, "Cannot establish reliable source video timing for split"
        )) from exc
    prefix, separator, suffix = dst_pattern.rpartition("%03d")
    if not separator:
        raise ToolError("Invalid internal split output pattern")
    files = [Path(f"{prefix}{index:03d}{suffix}") for index in range(len(markers) + 1)]
    validated = False
    try:
        for path, interval in zip(files, timeline.intervals, strict=True):
            args = [*_BASE, "-copyts", "-i", src, "-map", "0:v:0"]
            args += [
                "-vf", f"trim=start_frame={interval.first_frame}:"
                f"end_frame={interval.first_frame + interval.frame_count},setpts=PTS-STARTPTS",
                "-fps_mode", "passthrough", "-c:v", options["video_codec"][0],
                "-pix_fmt", "yuv420p",
            ]
            if metadata.has_audio_stream:
                args += [
                    "-map", "0:a:0", "-af",
                    f"asetpts=PTS-({timeline.start_seconds})/TB,"
                    f"atrim=start={interval.start_seconds}:end={interval.end_seconds},"
                    f"asetpts=PTS-({interval.start_seconds})/TB",
                    "-c:a", options["audio_codec"][0], "-b:a", "128k",
                ]
            else:
                args += ["-an"]
            await runner.run([*args, str(path)])
        measured_duration = 0.0
        for path, interval in zip(files, timeline.intervals, strict=True):
            if not path.is_file() or path.stat().st_size == 0:
                raise ToolError("Split output is missing or empty")
            output = await probe_media(path, MediaType.video)
            if not output.has_video_stream or not output.width or not output.height:
                raise ToolError("Split output does not contain valid video")
            actual = await probe_video_timeline(path, frame_rate=output.frame_rate)
            expected_duration = interval.end_seconds - interval.start_seconds
            tolerance = timeline.frame_tolerance + 0.002  # Matroska millisecond timebase.
            if (actual.intervals[0].frame_count != interval.frame_count
                    or abs(actual.duration_seconds - expected_duration) > tolerance):
                raise ToolError("Split output did not match the requested boundary")
            if metadata.has_audio_stream and not output.has_audio_stream:
                raise ToolError("Split output did not preserve the source soundtrack")
            measured_duration += actual.duration_seconds
        if abs(measured_duration - timeline.duration_seconds) > len(files) * tolerance:
            raise ToolError("Split output coverage did not match the source video")
        validated = True
        return files
    except MetadataProbeError as exc:
        raise ToolError("Cannot validate split output video timing") from exc
    finally:
        # The handler receives files only after complete validation. On any
        # failure/cancellation, remove this operation's files, not arbitrary glob matches.
        if not validated:
            for path in files:
                with suppress(OSError):
                    path.unlink(missing_ok=True)


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


async def probe_audio_duration(path: str) -> float:
    """Return the first audio stream duration, falling back to container duration."""
    out = await runner.run_stdout(
        [
            FFPROBE, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=duration:format=duration", "-of", "json", path,
        ],
        timeout=60,
    )
    try:
        payload = json.loads(out)
        duration = payload.get("streams", [{}])[0].get("duration")
        try:
            return float(duration)
        except (TypeError, ValueError):
            return float(payload.get("format", {}).get("duration", 0.0))
    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
        return 0.0


def _video_audio_codec(dst: str) -> tuple[str, str]:
    """Choose a known audio encoder for the preserved source container."""
    extension = Path(dst).suffix.lower()
    if extension in {".mp4", ".mov", ".mkv"}:
        return "aac", "192k"
    if extension == ".webm":
        return "libopus", "128k"
    raise ToolError(
        f"Audio processing is not supported for video container '{extension or 'unknown'}'"
    )


async def video_audio_transform(src: str, dst: str, audio_filter: str) -> None:
    """Transform the first audio stream while stream-copying the first video.

    Explicit mapping intentionally omits subtitle/data/other audio streams.
    The selected convention is first video + first audio stream by input order.
    """
    if not await _has_audio(src):
        raise ToolError("Cannot process soundtrack: video has no audio stream")
    audio_codec, bitrate = _video_audio_codec(dst)
    await runner.run(
        [
            *_BASE, "-i", src,
            "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "copy", "-c:a", audio_codec, "-b:a", bitrate,
            "-filter:a", audio_filter,
            dst,
        ]
    )


async def video_audio_extract(src: str, dst: str) -> None:
    """Extract the first audio stream as AAC in an M4A container."""
    if not await _has_audio(src):
        raise ToolError("Cannot extract audio: video has no audio stream")
    await runner.run(
        [
            *_BASE, "-i", src, "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-c:a", "aac", "-b:a", "192k", "-f", "ipod", dst,
        ]
    )


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


async def video_audio_gain(src: str, dst: str, gain_db: float) -> None:
    await video_audio_transform(
        src, dst, f"volume={_num(gain_db, 'gain_db')}dB"
    )


async def audio_normalize(src: str, dst: str, target_i: float) -> None:
    filt = f"loudnorm=I={_num(target_i, 'target_i')}:TP=-1.5:LRA=11"
    await runner.run([*_BASE, "-i", src, "-filter:a", filt, dst])


async def video_audio_normalize(src: str, dst: str, target_i: float) -> None:
    filt = f"loudnorm=I={_num(target_i, 'target_i')}:TP=-1.5:LRA=11"
    await video_audio_transform(src, dst, filt)


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


async def video_audio_fade(src: str, dst: str, fade_in: float, fade_out: float) -> None:
    filters: list[str] = []
    if fade_in:
        filters.append(f"afade=t=in:st=0:d={_num(fade_in, 'fade_in')}")
    if fade_out:
        duration = await probe_audio_duration(src)
        start = max(duration - float(fade_out), 0.0)
        filters.append(f"afade=t=out:st={start}:d={_num(fade_out, 'fade_out')}")
    if not filters:
        raise ToolError("audio.fade requires fade_in and/or fade_out > 0")
    await video_audio_transform(src, dst, ",".join(filters))


DEFAULT_RNNOISE_MODEL = "/usr/share/rnnoise/model.rnnn"


async def audio_denoise(src: str, dst: str, strength: float | None) -> None:
    mix = 1.0 if strength is None else strength
    filt = f"arnndn=m={DEFAULT_RNNOISE_MODEL}:mix={_num(mix, 'strength')}"
    await runner.run([*_BASE, "-i", src, "-filter:a", filt, dst])


async def video_audio_denoise(src: str, dst: str, strength: float | None) -> None:
    mix = 1.0 if strength is None else strength
    filt = f"arnndn=m={DEFAULT_RNNOISE_MODEL}:mix={_num(mix, 'strength')}"
    await video_audio_transform(src, dst, filt)


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


async def kenburns(
    image: str,
    dst: str,
    *,
    duration: float,
    zoom: float,
    direction: str,
    width: int,
    height: int,
    focus_x: float,
    focus_y: float,
) -> None:
    """Ken Burns pan-and-zoom: a still image -> a video clip zooming toward
    (focus_x, focus_y) (fractions of image width/height).

    Deliberately does NOT `-loop 1` the input: zoompan expands a single still
    frame into `d` output frames itself, and combining the two breaks the pan
    -- each looped copy arrives as a "new" input frame, which resets zoompan's
    internal zoom accumulator instead of advancing it (confirmed empirically:
    with -loop 1, zoom animates correctly but x/y stay frozen at their
    frame-0 values; without it, both animate together as expected).
    """
    _num(duration, "duration")
    _num(zoom, "zoom")
    if direction not in ("in", "out"):
        raise ToolError(f"direction must be 'in' or 'out', got {direction!r}")
    if zoom <= 1.0:
        raise ToolError(f"zoom must be greater than 1.0, got {zoom!r}")

    frames = max(round(duration * FPS), 1)
    step = (zoom - 1.0) / frames
    zoompan = (
        f"zoompan=z='min(zoom+{step},{zoom})':"
        f"x='{focus_x}*iw-(iw/zoom/2)':y='{focus_y}*ih-(ih/zoom/2)':"
        f"d={frames}:s={_num(width, 'width')}x{_num(height, 'height')}:fps={FPS}"
    )
    vf = zoompan if direction == "in" else f"{zoompan},reverse"

    await runner.run(
        [
            *_BASE, "-i", image, *_ANULL,
            "-vf", vf, "-map", "0:v", "-map", "1:a",
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


_LOWERTHIRD_POSITIONS = {
    "bottom-left": "20:main_h-overlay_h-20",
    "bottom-center": "(main_w-overlay_w)/2:main_h-overlay_h-20",
    "bottom-right": "main_w-overlay_w-20:main_h-overlay_h-20",
}


async def overlay_lowerthird(
    video: str,
    overlay_png: str,
    dst: str,
    *,
    start: float,
    duration: float,
    position: str = "bottom-left",
) -> None:
    """Overlay a (small, pre-rendered) graphic onto `video` for [start, start+duration)."""
    if position not in _LOWERTHIRD_POSITIONS:
        raise ToolError(
            f"position must be one of {sorted(_LOWERTHIRD_POSITIONS)}, got {position!r}"
        )
    _num(start, "start")
    _num(duration, "duration")
    xy = _LOWERTHIRD_POSITIONS[position]
    filt = f"[0:v][1:v]overlay={xy}:enable='between(t,{start},{start + duration})'[v]"
    await runner.run(
        [
            *_BASE, "-i", video, "-i", overlay_png,
            "-filter_complex", filt, "-map", "[v]", "-map", "0:a?",
            *_PIX, "-c:a", "copy", dst,
        ]
    )


def _compose_encode_args(policy: ComposeRenderPolicy) -> list[str]:
    args = [
        "-pix_fmt", policy.pixel_format.value,
        "-c:v", policy.video_codec.value,
        "-c:a", policy.audio_codec.value,
    ]
    if policy.audio_sample_format is not None:
        args.extend(["-sample_fmt", policy.audio_sample_format.value])
    return args


def _compose_audio_encode_args(policy: ComposeRenderPolicy | None) -> list[str]:
    if policy is None:
        return []
    args = ["-c:a", policy.audio_codec.value]
    if policy.audio_sample_format is not None:
        args.extend(["-sample_fmt", policy.audio_sample_format.value])
    return args


async def subtitle_embed(
    video: str,
    subtitle: str,
    dst: str,
    mode: str = "soft",
    *,
    compose_policy: ComposeRenderPolicy | None = None,
) -> None:
    if mode == "burn":
        args = [*_BASE, "-i", video, "-vf", f"subtitles='{_escape_subs(subtitle)}'"]
        if compose_policy is None:
            args.extend(_PIX)
        else:
            args.extend(["-map", "0:v:0", "-map", "0:a:0?"])
            args.extend(_compose_encode_args(compose_policy))
        args.append(dst)
        await runner.run(args)
    else:  # soft mux
        await runner.run(
            [*_BASE, "-i", video, "-i", subtitle, "-c", "copy", "-c:s", "mov_text", dst]
        )


async def audio_mix(
    media: str,
    music: str,
    dst: str,
    music_volume: float = 0.3,
    mode: str = "mix",
    *,
    compose_policy: ComposeRenderPolicy | None = None,
) -> None:
    """Lay `music` under `media`'s audio. mode: mix | duck (≈low mix) | replace."""
    vol = music_volume if mode != "duck" else min(music_volume, 0.2)
    has_audio = await _has_audio(media)
    if mode == "replace":
        await runner.run(
            [*_BASE, "-i", media, "-i", music,
             "-map", "0:v?", "-map", "1:a", "-c:v", "copy",
             *_compose_audio_encode_args(compose_policy), "-shortest", dst]
        )
        return
    if not has_audio:
        await runner.run(
            [*_BASE, "-i", media, "-i", music,
             "-filter_complex", f"[1:a]volume={vol}[a]",
             "-map", "0:v?", "-map", "[a]", "-c:v", "copy",
             *_compose_audio_encode_args(compose_policy), "-shortest", dst]
        )
        return
    filt = f"[1:a]volume={vol}[m];[0:a][m]amix=inputs=2:duration=first[a]"
    await runner.run(
        [*_BASE, "-i", media, "-i", music,
         "-filter_complex", filt, "-map", "0:v?", "-map", "[a]",
         "-c:v", "copy", *_compose_audio_encode_args(compose_policy), dst]
    )


async def _normalize_segment(
    src: str,
    dst: str,
    width: int,
    height: int,
    policy: ComposeRenderPolicy = DEFAULT_COMPOSE_RENDER_POLICY,
) -> None:
    """Normalize first video/soundtrack; synthesize silence only if absent."""
    duration = await _compose_video_duration(src)
    has_audio = await _has_audio(src)
    audio_input = [] if has_audio else _ANULL
    audio_stream = "0:a:0" if has_audio else "1:a:0"
    await runner.run(
        [
            *_BASE, "-i", src, *audio_input,
            "-map", "0:v:0", "-map", audio_stream,
            "-vf", _scale_pad(width, height),
            # Pad short soundtracks and retain timestamp gaps. Bound both real
            # and synthetic audio to video, not container/audio duration, so
            # concat cannot insert a trailing audio-only gap between clips.
            "-af", f"aresample=44100:async=1:first_pts=0,apad,atrim=duration={duration}",
            "-ar", "44100", "-ac", "2",
            *_compose_encode_args(policy), "-t", str(duration), dst,
        ]
    )


async def _compose_video_duration(src: str) -> float:
    """Use the first video's duration (Matroska stores it as a stream tag)."""
    out = await runner.run_stdout(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration:stream_tags=DURATION:format=duration", "-of", "json", src],
        timeout=60,
    )
    try:
        payload = json.loads(out)
        stream = payload["streams"][0]
        value = stream.get("duration")
        if value in (None, "N/A"):
            tag = stream.get("tags", {}).get("DURATION")
            if tag is not None:
                hours, minutes, seconds = map(float, tag.split(":"))
                value = hours * 3600 + minutes * 60 + seconds
            else:
                value = payload["format"]["duration"]
        duration = float(value)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError
        return duration
    except (
        json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError, AttributeError,
    ) as exc:
        raise ToolError("Cannot determine composition source video duration") from exc


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
    policy: ComposeRenderPolicy = DEFAULT_COMPOSE_RENDER_POLICY,
) -> None:
    """Timeline render with one trusted encoding policy for every media stage."""
    if Path(dst).suffix.lower() != policy.suffix:
        raise ToolError(
            f"Compose output suffix must be '{policy.suffix}' for this render policy"
        )

    # 1. normalize each segment to identical codec/params
    normalized: list[Path] = []
    for i, seg in enumerate(segments):
        out = work_dir / f"_norm_{i:03d}{policy.suffix}"
        await _normalize_segment(seg, str(out), width, height, policy)
        normalized.append(out)

    # 2. concat via the demuxer (copy; params already identical)
    listing = work_dir / "_concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in normalized), encoding="utf-8")
    concat = work_dir / f"_concat{policy.suffix}"
    await runner.run(
        [*_BASE, "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(concat)]
    )

    current = concat
    # 3. optional audio bed over the whole timeline
    if audio_path:
        with_audio = work_dir / f"_audio{policy.suffix}"
        await audio_mix(str(current), audio_path, str(with_audio),
                        music_volume=music_volume, mode=audio_mode,
                        compose_policy=policy)
        current = with_audio

    # 4. optional subtitle burn-in (final re-encode)
    if subtitle_path:
        burned = work_dir / f"_subs{policy.suffix}"
        await subtitle_embed(
            str(current), subtitle_path, str(burned), mode="burn",
            compose_policy=policy,
        )
        current = burned

    # 5. move final result to dst
    if str(current) != dst:
        await runner.run([*_BASE, "-i", str(current), "-c", "copy", dst])


async def export_delivery_transform(
    src: str, dst: str, settings: EffectiveExportSettings
) -> None:
    """Encode a validated lossy delivery profile without accepting CLI fragments."""
    if settings.composition_policy != CompositionPolicyId.DEFAULT:
        raise ToolError("Lossless archival settings must render directly through compose")

    args = [*_BASE, "-i", src, "-map", "0:v:0", "-map", "0:a:0?"]
    video_filter: str | None = None
    if settings.target_width is not None and settings.target_height is not None:
        if settings.fit != FitPolicy.PAD:
            raise ToolError("Exact delivery canvases require the preserve-and-pad fit policy")
        width, height = settings.target_width, settings.target_height
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
            f"force_divisible_by=2,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    elif settings.max_width is not None and settings.max_height is not None:
        if settings.fit != FitPolicy.FIT:
            raise ToolError("Maximum delivery bounds require the aspect-preserving fit policy")
        video_filter = (
            f"scale=w='min(iw,{settings.max_width})':h='min(ih,{settings.max_height})':"
            "force_original_aspect_ratio=decrease:force_divisible_by=2"
        )
    if video_filter is not None:
        args.extend(["-vf", video_filter])

    args.extend(
        [
            "-c:v", settings.video_codec.value,
            "-pix_fmt", settings.pixel_format.value,
        ]
    )
    if settings.profile is not None:
        args.extend(["-profile:v", settings.profile])
    if settings.crf is not None:
        args.extend(["-crf", str(settings.crf)])
    args.extend(["-c:a", settings.audio_codec.value])
    if settings.audio_bitrate_kbps is not None:
        args.extend(["-b:a", f"{settings.audio_bitrate_kbps}k"])
    if settings.audio_channels is not None:
        args.extend(["-ac", str(settings.audio_channels)])
    if settings.faststart:
        if settings.container != ExportContainer.MP4:
            raise ToolError("Faststart is supported only for MP4 delivery")
        args.extend(["-movflags", "+faststart"])
    args.extend(["-f", "mp4" if settings.container == ExportContainer.MP4 else "matroska", dst])
    await runner.run(args)
