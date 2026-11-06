"""FFmpeg wrappers for deterministic video and audio operations.

Each function builds an argument list and delegates to runner.run. Output
container/codec is generally chosen by FFmpeg from the output file extension.
"""

from __future__ import annotations

import dataclasses
import json
import math
from contextlib import suppress
from pathlib import Path

from app.capabilities.media_options import VIDEO_QUALITY_CRF, VIDEO_TRANSCODE_OPTIONS
from app.core.errors import ToolError, ValidationError
from app.models.enums import MediaType
from app.services.media_metadata import MetadataProbeError, probe_media, probe_video_timeline
from app.tools import progress, runner
from app.tools.compose_policy import (
    DEFAULT_COMPOSE_RENDER_POLICY,
    LOSSLESS_COMPOSE_RENDER_POLICY,
    ComposeRenderPolicy,
)
from app.tools.export_policy import (
    CompositionPolicyId,
    EffectiveExportSettings,
    ExportContainer,
    FitPolicy,
)
from app.tools.geometry import check_crop_rectangle, is_quarter_turn

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


def resolve_trim_window(
    start: float,
    end: float | None,
    *,
    duration: float | None,
    frame_rate: float | None,
    media: str = "video",
) -> float | None:
    """Validate a trim interval against known source timing; return the effective end.

    ``None`` means "until the end of the video": an omitted end, or one at/after
    the known duration. Unknown duration/frame rate skip only the checks that
    need them; execution validates the rendered output independently.
    """
    if start < 0:
        raise ValidationError("Trim start must not be negative")
    if end is not None and end <= start:
        raise ValidationError("Trim end must be after the start")
    if duration is not None and duration > 0:
        if start >= duration:
            raise ValidationError(
                f"Trim start must be before the end of the {media} ({duration:.3f} s)"
            )
        if end is not None and end >= duration:
            end = None
    if frame_rate is not None and frame_rate > 0:
        stop = end if end is not None else duration
        if stop is not None and stop - start < 1 / frame_rate - 1e-6:
            raise ValidationError("Trim interval must be at least one video frame long")
    return end


async def video_trim(src: str, dst: str, start: float, end: float | None = None) -> None:
    """Keep [start, end) (or start onward) and publish only a frame-bearing output."""
    _num(start, "start")
    if end is not None:
        _num(end, "end")
    try:
        metadata = await probe_media(Path(src), MediaType.video)
    except MetadataProbeError as exc:
        raise ToolError("Cannot read source video timing for trim") from exc
    if not metadata.has_video_stream:
        raise ValidationError("Trim source does not contain a video stream")
    effective_end = resolve_trim_window(
        start, end, duration=metadata.duration_seconds, frame_rate=metadata.frame_rate
    )
    args = [*_BASE, "-i", src, "-ss", str(start)]
    if effective_end is not None:
        args += ["-to", str(effective_end)]
    stop = effective_end if effective_end is not None else metadata.duration_seconds
    if stop is not None:
        progress.expect_output_seconds(stop - start)
    output = Path(dst)
    validated = False
    try:
        await runner.run([*args, dst])
        if not output.is_file() or output.stat().st_size == 0:
            raise ToolError("Trim output is missing or empty")
        try:
            result = await probe_media(output, MediaType.video)
            if not result.has_video_stream:
                raise ToolError("Trim output does not contain valid video")
            await probe_video_timeline(output, frame_rate=result.frame_rate)
        except MetadataProbeError as exc:
            raise ToolError("Trim output does not contain video frames") from exc
        validated = True
    finally:
        if not validated:
            with suppress(OSError):
                output.unlink()


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


@dataclasses.dataclass(frozen=True)
class AudioFormat:
    """First audio stream's format, used to keep processed audio like its source."""

    codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    sample_fmt: str | None = None
    bit_rate: int | None = None


def _positive_int(value: object) -> int | None:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


async def probe_audio_format(path: str) -> AudioFormat:
    """Probe the first audio stream; unknown fields stay None (never fails)."""
    try:
        out = await runner.run_stdout(
            [
                FFPROBE, "-v", "error", "-select_streams", "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels,sample_fmt,bit_rate:format=bit_rate",
                "-of", "json", path,
            ],
            timeout=60,
        )
        payload = json.loads(out)
        stream = payload.get("streams", [{}])[0]
    except (ToolError, json.JSONDecodeError, IndexError, AttributeError, TypeError):
        return AudioFormat()
    if not isinstance(stream, dict):
        return AudioFormat()
    bit_rate = _positive_int(stream.get("bit_rate"))
    codec = stream.get("codec_name") if isinstance(stream.get("codec_name"), str) else None
    if bit_rate is None and codec is not None and not _is_lossless_audio(codec):
        # Audio-only containers often report the rate only at format level.
        audio_only = len(payload.get("streams", [])) == 1
        fmt = payload.get("format", {})
        if audio_only and isinstance(fmt, dict):
            bit_rate = _positive_int(fmt.get("bit_rate"))
    sample_fmt = stream.get("sample_fmt")
    return AudioFormat(
        codec=codec,
        sample_rate=_positive_int(stream.get("sample_rate")),
        channels=_positive_int(stream.get("channels")),
        sample_fmt=sample_fmt if isinstance(sample_fmt, str) else None,
        bit_rate=bit_rate,
    )


_LOSSY_AUDIO_ENCODERS = {
    "mp3": "libmp3lame",
    "aac": "aac",
    "vorbis": "libvorbis",
    "opus": "libopus",
}
_OPUS_SAMPLE_RATES = {8000, 12000, 16000, 24000, 48000}
# Bounds for re-encoding a video soundtrack at its source bitrate.
_SOUNDTRACK_MIN_BIT_RATE = 32_000
_SOUNDTRACK_MAX_BIT_RATE = 320_000


def _is_lossless_audio(codec: str) -> bool:
    return codec.startswith("pcm_") or codec in {"flac", "alac"}


def _rate_and_layout_args(source: AudioFormat, encoder: str | None) -> list[str]:
    args: list[str] = []
    if source.sample_rate is not None:
        if encoder == "libopus" and source.sample_rate not in _OPUS_SAMPLE_RATES:
            args += ["-ar", "48000"]
        else:
            args += ["-ar", str(source.sample_rate)]
    if source.channels is not None:
        args += ["-ac", str(source.channels)]
    return args


def preserved_audio_args(source: AudioFormat) -> list[str]:
    """Encode an audio-only result like its source.

    Filters such as loudnorm (192 kHz) or arnndn (48 kHz) resample internally;
    without this a 16-bit 44.1 kHz WAV came back several times larger.
    """
    codec = source.codec
    if codec is None:
        return _rate_and_layout_args(source, None)
    if _is_lossless_audio(codec):
        args = ["-c:a", codec]
        if not codec.startswith("pcm_") and source.sample_fmt:
            args += ["-sample_fmt", source.sample_fmt]
        return [*args, *_rate_and_layout_args(source, codec)]
    encoder = _LOSSY_AUDIO_ENCODERS.get(codec)
    if encoder is None:
        return _rate_and_layout_args(source, None)
    args = ["-c:a", encoder]
    if source.bit_rate is not None:
        args += ["-b:a", str(source.bit_rate)]
    return [*args, *_rate_and_layout_args(source, encoder)]


async def _run_audio_filter(src: str, dst: str, audio_filter: str) -> None:
    source = await probe_audio_format(src)
    await runner.run(
        [*_BASE, "-i", src, "-filter:a", audio_filter, *preserved_audio_args(source), dst]
    )


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
    source = await probe_audio_format(src)
    if source.bit_rate is not None:
        # Keep the soundtrack's size: re-encoding a 64k track at 192k grew videos.
        bitrate = str(
            min(max(source.bit_rate, _SOUNDTRACK_MIN_BIT_RATE), _SOUNDTRACK_MAX_BIT_RATE)
        )
    await runner.run(
        [
            *_BASE, "-i", src,
            "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "copy", "-c:a", audio_codec, "-b:a", bitrate,
            *_rate_and_layout_args(source, audio_codec),
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


async def _concat_normalize(
    src: str, dst: str, width: int, height: int, *, video_codec: str, audio_codec: str,
) -> None:
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
        "-ar", "48000" if audio_codec == "libopus" else "44100", "-ac", "2",
        *_PIX, "-c:v", video_codec, "-c:a", audio_codec, "-shortest", dst,
    ]
    await runner.run(args)


async def video_concat(sources: list[str], dst: str, work_dir: Path) -> None:
    """Concatenate video clips, normalizing each to the first clip's resolution.

    The ffmpeg `concat` filter/demuxer require identical stream parameters
    across inputs, so each clip is first scaled/padded to a common size and
    re-encoded to uniform codecs before a stream-copy concat.
    """
    container = Path(dst).suffix.lstrip(".").lower()
    options = VIDEO_TRANSCODE_OPTIONS.get(container)
    if options is None:
        raise ValidationError("Video concat does not support this output container")
    video_codec, audio_codec = options["video_codec"][0], options["audio_codec"][0]
    normalized = [work_dir / f"_concat_norm_{i:03d}.{container}" for i in range(len(sources))]
    listing = work_dir / "_concat_list.txt"
    output = Path(dst)
    validated = False
    try:
        width, height = await probe_video_size(sources[0])
        if width <= 0 or height <= 0:
            width, height = 1920, 1080
        for src, out in zip(sources, normalized, strict=True):
            await _concat_normalize(
                src, str(out), width, height, video_codec=video_codec, audio_codec=audio_codec,
            )
        listing.write_text("".join(f"file '{p}'\n" for p in normalized), encoding="utf-8")
        await runner.run(
            [*_BASE, "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", dst]
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise ToolError("Concat output is missing or empty")
        metadata = await probe_media(output, MediaType.video)
        # ffprobe reports codec IDs, not FFmpeg encoder implementation names.
        expected_video = {"libx264": "h264", "libvpx-vp9": "vp9"}[video_codec]
        expected_container = "matroska" if container == "mkv" else container
        if (not metadata.has_video_stream or not metadata.width or not metadata.height
                or not metadata.has_audio_stream or not metadata.duration_seconds
                or metadata.duration_seconds <= 0
                or expected_container not in (metadata.container_format or "").split(",")
                or metadata.video_codec != expected_video
                or metadata.audio_codec != audio_codec.removeprefix("lib")):
            raise ToolError("Concat output does not match the destination media policy")
        validated = True
    except MetadataProbeError as exc:
        raise ToolError("Concat output could not be validated") from exc
    finally:
        for path in [*normalized, listing, *([] if validated else [output])]:
            with suppress(OSError):
                path.unlink(missing_ok=True)


# "compressed" promises a smaller file: its constant-quality target is capped at
# this share of the source video bitrate, because an already efficiently encoded
# source otherwise came out larger (devfixes C-4: +10% and +28% on real 1080p).
COMPRESSED_MAX_SOURCE_BIT_RATE_RATIO = 0.75


async def probe_video_bit_rate(path: str) -> int | None:
    """First video stream's bitrate, else container rate minus audio; None if unknown."""
    try:
        out = await runner.run_stdout(
            [
                FFPROBE, "-v", "error", "-show_entries",
                "stream=codec_type,bit_rate:format=bit_rate", "-of", "json", path,
            ],
            timeout=60,
        )
        payload = json.loads(out)
        streams = [s for s in payload.get("streams", []) if isinstance(s, dict)]
        fmt = payload.get("format", {})
    except (ToolError, json.JSONDecodeError, AttributeError, TypeError):
        return None
    video = [s for s in streams if s.get("codec_type") == "video"]
    if not video:
        return None
    rate = _positive_int(video[0].get("bit_rate"))
    if rate is not None:
        return rate
    total = _positive_int(fmt.get("bit_rate")) if isinstance(fmt, dict) else None
    if total is None:
        return None
    others = 0
    for stream in streams:
        if stream is video[0]:
            continue
        other = _positive_int(stream.get("bit_rate"))
        if other is None and stream.get("codec_type") in {"audio", "video"}:
            return None
        others += other or 0
    return _positive_int(total - others)


async def video_transcode(
    src: str, dst: str, video_codec: str | None, audio_codec: str | None,
    *, quality: str | None = None,
) -> int | None:
    """Transcode; returns the video bitrate cap applied for "compressed", if any."""
    args = [*_BASE, "-i", src]
    if video_codec:
        args += ["-c:v", video_codec]
    if audio_codec:
        args += ["-c:a", audio_codec]
    cap: int | None = None
    if quality is not None:
        levels = VIDEO_QUALITY_CRF.get(video_codec or "", {})
        if quality not in levels:
            raise ToolError("Unsupported video quality/codec combination.")
        args += ["-crf", str(levels[quality])]
        if quality == "compressed":
            source_rate = await probe_video_bit_rate(src)
            if source_rate is not None:
                cap = max(int(source_rate * COMPRESSED_MAX_SOURCE_BIT_RATE_RATIO), 1000)
        if video_codec == "libvpx-vp9":
            # Constant quality, or constrained quality under the cap.
            args += ["-b:v", str(cap) if cap is not None else "0"]
        elif cap is not None:
            args += ["-maxrate", str(cap), "-bufsize", str(cap)]
    args.append(dst)
    await runner.run(args)
    return cap


async def video_mute(src: str, dst: str) -> None:
    await runner.run([*_BASE, "-i", src, "-an", "-c:v", "copy", dst])


async def probe_display_size(path: str) -> tuple[int, int]:
    """Return the first video stream's size as displayed, i.e. after rotation.

    FFmpeg auto-rotates before filters, so crop coordinates refer to this size,
    not the coded width/height stored in Asset technical metadata.
    """
    out = await runner.run_stdout([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:stream_side_data=rotation:stream_tags=rotate",
        "-of", "json", path,
    ])
    try:
        stream = json.loads(out)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ToolError("Cannot read source video dimensions") from exc
    rotation: object = next(
        (entry["rotation"] for entry in stream.get("side_data_list", [])
         if isinstance(entry, dict) and "rotation" in entry),
        stream.get("tags", {}).get("rotate", 0),
    )
    return (height, width) if is_quarter_turn(rotation) else (width, height)


async def video_crop(src: str, dst: str, x: int, y: int, width: int, height: int) -> None:
    crop = f"crop={_num(width, 'width')}:{_num(height, 'height')}:{_num(x, 'x')}:{_num(y, 'y')}"
    frame_width, frame_height = await probe_display_size(src)
    check_crop_rectangle(x, y, width, height, frame_width=frame_width, frame_height=frame_height)
    output = Path(dst)
    validated = False
    try:
        await runner.run([*_BASE, "-i", src, "-vf", crop, dst])
        if not output.is_file() or output.stat().st_size == 0:
            raise ToolError("Crop output is missing or empty")
        validated = True
    finally:
        if not validated:
            with suppress(OSError):
                output.unlink()


def even_scale_dimensions(width: int | None, height: int | None) -> tuple[int, int]:
    """Map requested resize dimensions to an FFmpeg scale pair with even sizes.

    4:2:0 encoders (H.264, VP9 as used here) reject odd sizes, so explicit sizes
    round down to even and an omitted side becomes -2: aspect-preserving,
    rounded to even. Shared by the direct handler and the Agent render path.
    """
    if width is None and height is None:
        raise ValidationError("Resize needs a width, a height, or both")
    sides = []
    for name, value in (("width", width), ("height", height)):
        if value is None:
            sides.append(-2)
            continue
        _num(value, name)
        if value < 2:
            raise ValidationError(f"Resize {name} must be at least 2 pixels")
        sides.append(value - value % 2)
    return sides[0], sides[1]


async def video_resize(
    src: str, dst: str, width: int | None = None, height: int | None = None
) -> None:
    out_width, out_height = even_scale_dimensions(width, height)
    await runner.run([*_BASE, "-i", src, "-vf", f"scale={out_width}:{out_height}", dst])


def check_thumbnail_time(timestamp: float, *, duration: float | None) -> None:
    """Reject a frame time outside the video; unknown duration defers to execution."""
    if timestamp < 0:
        raise ValidationError("Thumbnail time must not be negative")
    if duration is not None and duration > 0 and timestamp >= duration:
        raise ValidationError(
            f"Thumbnail time must be before the end of the video ({duration:.3f} s)"
        )


async def video_thumbnail(src: str, dst: str, timestamp: float) -> None:
    """Extract the frame displayed at ``timestamp`` and publish only a real image."""
    _num(timestamp, "timestamp")
    try:
        metadata = await probe_media(Path(src), MediaType.video)
    except MetadataProbeError as exc:
        raise ToolError("Cannot read source video timing for thumbnail") from exc
    if not metadata.has_video_stream:
        raise ValidationError("Thumbnail source does not contain a video stream")
    check_thumbnail_time(timestamp, duration=metadata.duration_seconds)
    output = Path(dst)
    validated = False
    try:
        await runner.run([*_BASE, "-ss", str(timestamp), "-i", src, "-frames:v", "1", dst])
        if not output.is_file() or output.stat().st_size == 0:
            # A time inside the duration but after the last frame's timestamp
            # still shows that last frame; FFmpeg's input seek writes nothing.
            await runner.run([*_BASE, "-sseof", "-1", "-i", src, "-update", "1", dst])
        if not output.is_file() or output.stat().st_size == 0:
            raise ToolError("Thumbnail extraction produced no image")
        validated = True
    finally:
        if not validated:
            with suppress(OSError):
                output.unlink()


# --- audio ---------------------------------------------------------------


async def audio_trim(src: str, dst: str, start: float, end: float | None = None) -> None:
    """Keep [start, end) (or start onward) and publish only output with audio."""
    _num(start, "start")
    if end is not None:
        _num(end, "end")
    try:
        metadata = await probe_media(Path(src), MediaType.audio)
    except MetadataProbeError as exc:
        raise ToolError("Cannot read source audio timing for trim") from exc
    if not metadata.has_audio_stream:
        raise ValidationError("Trim source does not contain an audio stream")
    effective_end = resolve_trim_window(
        start, end, duration=metadata.duration_seconds, frame_rate=None, media="audio",
    )
    args = [*_BASE, "-i", src, "-ss", str(start)]
    if effective_end is not None:
        args += ["-to", str(effective_end)]
    stop = effective_end if effective_end is not None else metadata.duration_seconds
    if stop is not None:
        progress.expect_output_seconds(stop - start)
    output = Path(dst)
    validated = False
    try:
        await runner.run([*args, dst])
        if not output.is_file() or output.stat().st_size == 0:
            raise ToolError("Trim output is missing or empty")
        try:
            result = await probe_media(output, MediaType.audio)
        except MetadataProbeError as exc:
            raise ToolError("Trim output could not be read") from exc
        if not result.has_audio_stream or not result.duration_seconds:
            raise ToolError("Trim output does not contain audio")
        validated = True
    finally:
        if not validated:
            with suppress(OSError):
                output.unlink()


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
    await _run_audio_filter(src, dst, f"volume={_num(gain_db, 'gain_db')}dB")


async def video_audio_gain(src: str, dst: str, gain_db: float) -> None:
    await video_audio_transform(
        src, dst, f"volume={_num(gain_db, 'gain_db')}dB"
    )


async def audio_normalize(src: str, dst: str, target_i: float) -> None:
    filt = f"loudnorm=I={_num(target_i, 'target_i')}:TP=-1.5:LRA=11"
    await _run_audio_filter(src, dst, filt)


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
    await _run_audio_filter(src, dst, ",".join(filters))


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
    await _run_audio_filter(src, dst, filt)


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
    apply_replace_gain: bool,
    compose_policy: ComposeRenderPolicy | None = None,
) -> None:
    """Mix/duck/replace audio; callers explicitly choose replacement gain semantics."""
    vol = music_volume if mode != "duck" else min(music_volume, 0.2)
    has_audio = await _has_audio(media)
    if mode == "replace":
        replacement = (
            ["-filter_complex", f"[1:a]volume={vol}[a]", "-map", "[a]"]
            if apply_replace_gain else ["-map", "1:a"]
        )
        await runner.run(
            [*_BASE, "-i", media, "-i", music,
             "-map", "0:v?", *replacement, "-c:v", "copy",
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
                        # Composition replacement beds retain their unscaled contract.
                        apply_replace_gain=False,
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


async def video_export(src: str, dst: str, settings: EffectiveExportSettings) -> None:
    """Render one video to a named delivery preset; publish only valid video."""
    output = Path(dst)
    validated = False
    try:
        if settings.composition_policy == CompositionPolicyId.LOSSLESS:
            # The same locked FFV1/FLAC policy project.export's Archival uses.
            await runner.run([
                *_BASE, "-i", src, "-map", "0:v:0", "-map", "0:a:0?",
                *_compose_encode_args(LOSSLESS_COMPOSE_RENDER_POLICY), "-f", "matroska", dst,
            ])
        else:
            # Quality-targeted like project.export, with no size cap: an already
            # efficient source can come out larger, which size_comparison reports.
            await export_delivery_transform(src, dst, settings)
        if not output.is_file() or output.stat().st_size == 0:
            raise ToolError("Export output is missing or empty")
        try:
            result = await probe_media(output, MediaType.video)
        except MetadataProbeError as exc:
            raise ToolError("Export output could not be read") from exc
        if not result.has_video_stream:
            raise ToolError("Export output does not contain valid video")
        validated = True
    finally:
        if not validated:
            with suppress(OSError):
                output.unlink()


async def extract_speech_audio(src: str, dst: str) -> None:
    """First soundtrack as small mono AAC/M4A for speech recognition uploads.

    Keeps uploads well under provider size limits and avoids sending video
    containers some transcription endpoints reject.
    """
    try:
        metadata = await probe_media(Path(src), MediaType.video)
    except MetadataProbeError as exc:
        raise ToolError("Cannot read the input's audio") from exc
    if not metadata.has_audio_stream:
        raise ValidationError("The input has no sound to create subtitles from")
    output = Path(dst)
    validated = False
    try:
        await runner.run([
            *_BASE, "-i", src, "-map", "0:a:0", "-vn", "-ac", "1",
            "-c:a", "aac", "-b:a", "64k", dst,
        ])
        if not output.is_file() or output.stat().st_size == 0:
            raise ToolError("Speech audio extraction produced no output")
        validated = True
    finally:
        if not validated:
            with suppress(OSError):
                output.unlink()
