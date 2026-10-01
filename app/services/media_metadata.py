"""Bounded, application-owned technical metadata probing for media Assets."""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.asset import Asset
from app.models.enums import MediaType, TechnicalMetadataStatus
from app.services.storage import get_storage

logger = get_logger(__name__)

PROBE_TIMEOUT_SECONDS = 60.0
MAX_PROBE_OUTPUT_BYTES = 1024 * 1024
MAX_STREAMS = 64
MAX_TEXT_LENGTH = 128


class MetadataProbeError(Exception):
    """A safe, non-diagnostic probe failure code (never raw tool stderr)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class TechnicalMetadata:
    status: TechnicalMetadataStatus
    duration_seconds: float | None = None
    container_format: str | None = None
    width: int | None = None
    height: int | None = None
    has_video_stream: bool | None = None
    has_audio_stream: bool | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    frame_rate: float | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    bit_rate: int | None = None
    technical_streams: dict[str, Any] | None = None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result[:MAX_TEXT_LENGTH] or None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number > 0 else None


def _nonnegative_integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _frame_rate(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip() or value == "0/0":
        return None
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


async def _read_bounded_stdout(stream: asyncio.StreamReader) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > MAX_PROBE_OUTPUT_BYTES:
            raise MetadataProbeError("output_too_large")
        chunks.append(chunk)


async def _run_probe(args: list[str]) -> bytes:
    """Run a fixed-argument probe with timeout and bounded, stderr-free output."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as exc:
        raise MetadataProbeError("tool_unavailable") from exc

    assert proc.stdout is not None
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_SECONDS):
            output = await _read_bounded_stdout(proc.stdout)
            return_code = await proc.wait()
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise MetadataProbeError("timeout") from exc
    except BaseException:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise

    if return_code != 0:
        raise MetadataProbeError("tool_failed")
    return output


def _unavailable() -> TechnicalMetadata:
    return TechnicalMetadata(status=TechnicalMetadataStatus.unavailable)


async def _probe_image(path: Path) -> TechnicalMetadata:
    # The first frame/page is the representative image for multi-page formats.
    output = await _run_probe(["identify", "-ping", "-format", "%m\t%w\t%h", f"{path}[0]"])
    try:
        fmt, raw_width, raw_height = output.decode("utf-8").strip().split("\t")
        width, height = int(raw_width), int(raw_height)
        if width <= 0 or height <= 0:
            raise ValueError("invalid dimensions")
    except (UnicodeDecodeError, ValueError) as exc:
        raise MetadataProbeError("invalid_image_metadata") from exc
    return TechnicalMetadata(
        status=TechnicalMetadataStatus.available,
        container_format=_text(fmt),
        width=width,
        height=height,
        has_video_stream=False,
        has_audio_stream=False,
        technical_streams={
            "streams": [
                {
                    "index": 0,
                    "type": "image",
                    "codec": _text(fmt),
                    "width": width,
                    "height": height,
                }
            ],
            "streams_truncated": False,
        },
    )


def _normalized_stream(stream: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    simple_fields = {
        "index": _nonnegative_integer(stream.get("index")),
        "type": _text(stream.get("codec_type")),
        "codec": _text(stream.get("codec_name")),
        "profile": _text(stream.get("profile")),
        "width": _integer(stream.get("width")),
        "height": _integer(stream.get("height")),
        "frame_rate": _frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "frame_rate_raw": _text(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "sample_rate": _integer(stream.get("sample_rate")),
        "channels": _integer(stream.get("channels")),
        "channel_layout": _text(stream.get("channel_layout")),
        "bit_rate": _integer(stream.get("bit_rate")),
        "duration_seconds": _number(stream.get("duration")),
    }
    for key, value in simple_fields.items():
        if value is not None:
            result[key] = value

    tags = stream.get("tags")
    if isinstance(tags, dict) and (language := _text(tags.get("language"))):
        result["language"] = language
    disposition = stream.get("disposition")
    if isinstance(disposition, dict):
        flags = {name: bool(disposition.get(name, 0)) for name in ("default", "forced")}
        if any(flags.values()):
            result["disposition"] = flags
    return result


def _parse_ffprobe(payload: bytes) -> TechnicalMetadata:
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataProbeError("invalid_probe_output") from exc
    if not isinstance(raw, dict):
        raise MetadataProbeError("invalid_probe_output")

    raw_streams = raw.get("streams")
    if not isinstance(raw_streams, list):
        raw_streams = []
    truncated = len(raw_streams) > MAX_STREAMS
    streams = [
        _normalized_stream(item) for item in raw_streams[:MAX_STREAMS] if isinstance(item, dict)
    ]
    if not raw_streams:
        raise MetadataProbeError("no_stream_information")
    raw_video = next(
        (
            item
            for item in raw_streams
            if isinstance(item, dict) and item.get("codec_type") == "video"
        ),
        None,
    )
    raw_audio = next(
        (
            item
            for item in raw_streams
            if isinstance(item, dict) and item.get("codec_type") == "audio"
        ),
        None,
    )
    primary_video = _normalized_stream(raw_video) if raw_video else {}
    primary_audio = _normalized_stream(raw_audio) if raw_audio else {}

    raw_format = raw.get("format")
    if not isinstance(raw_format, dict):
        raw_format = {}
    duration = _number(raw_format.get("duration"))
    if duration is None:
        durations = [s["duration_seconds"] for s in streams if "duration_seconds" in s]
        duration = max(durations) if durations else None
    bit_rate = _integer(raw_format.get("bit_rate"))
    if bit_rate is None:
        bit_rate = _integer(primary_video.get("bit_rate")) or _integer(
            primary_audio.get("bit_rate")
        )

    return TechnicalMetadata(
        status=TechnicalMetadataStatus.available,
        duration_seconds=duration,
        container_format=_text(raw_format.get("format_name")),
        width=primary_video.get("width"),
        height=primary_video.get("height"),
        has_video_stream=raw_video is not None,
        has_audio_stream=raw_audio is not None,
        video_codec=primary_video.get("codec"),
        audio_codec=primary_audio.get("codec"),
        frame_rate=primary_video.get("frame_rate"),
        audio_sample_rate=primary_audio.get("sample_rate"),
        audio_channels=primary_audio.get("channels"),
        bit_rate=bit_rate,
        technical_streams={"streams": streams, "streams_truncated": truncated},
    )


async def probe_media(path: Path, media_type: MediaType) -> TechnicalMetadata:
    """Probe supported media once and return a bounded normalized summary."""
    if not path.is_file():
        raise MetadataProbeError("file_missing")
    if media_type == MediaType.image:
        return await _probe_image(path)
    if media_type not in (MediaType.audio, MediaType.video):
        return _unavailable()

    entries = (
        "format=format_name,duration,bit_rate:"
        "stream=index,codec_type,codec_name,profile,width,height,avg_frame_rate,"
        "r_frame_rate,sample_rate,channels,channel_layout,bit_rate,duration:"
        "stream_tags=language:stream_disposition=default,forced"
    )
    output = await _run_probe(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            entries,
            "-of",
            "json",
            str(path),
        ]
    )
    return _parse_ffprobe(output)


async def probe_media_safely(
    path: Path, media_type: MediaType, *, asset_id: uuid.UUID | None = None
) -> TechnicalMetadata:
    """Return an explicit unavailable result instead of making an Asset fail."""
    try:
        return await probe_media(path, media_type)
    except MetadataProbeError as exc:
        if asset_id is not None:
            logger.warning("Metadata probe unavailable for asset %s (%s)", asset_id, exc.reason)
        return _unavailable()
    except Exception as exc:  # noqa: BLE001 - probing must not make Assets unreadable
        if asset_id is not None:
            logger.warning(
                "Metadata probe unavailable for asset %s (%s)", asset_id, type(exc).__name__
            )
        return _unavailable()


def apply_metadata(asset: Asset, metadata: TechnicalMetadata) -> None:
    asset.technical_metadata_status = metadata.status.value
    asset.duration_seconds = metadata.duration_seconds
    asset.container_format = metadata.container_format
    asset.width = metadata.width
    asset.height = metadata.height
    asset.has_video_stream = metadata.has_video_stream
    asset.has_audio_stream = metadata.has_audio_stream
    asset.video_codec = metadata.video_codec
    asset.audio_codec = metadata.audio_codec
    asset.frame_rate = metadata.frame_rate
    asset.audio_sample_rate = metadata.audio_sample_rate
    asset.audio_channels = metadata.audio_channels
    asset.bit_rate = metadata.bit_rate
    asset.technical_streams = metadata.technical_streams


async def probe_asset_metadata(
    session: AsyncSession, asset_id: uuid.UUID
) -> TechnicalMetadataStatus | None:
    """Probe and persist one existing non-deleted Asset; safe to retry."""
    asset = await session.get(Asset, asset_id)
    if asset is None or asset.deleted_at is not None:
        return None
    if asset.technical_metadata_status == TechnicalMetadataStatus.available.value:
        return TechnicalMetadataStatus.available

    path = get_storage().get_path(asset.storage_path)
    metadata = await probe_media_safely(path, asset.media_type, asset_id=asset.id)

    if asset.deleted_at is not None:
        return None
    apply_metadata(asset, metadata)
    await session.commit()
    await session.refresh(asset)
    return metadata.status
