"""Trusted encoding policies for the shared composition renderer.

These policies describe renderer-owned behavior, not user-selectable FFmpeg
options. Composition semantics remain in ``ComposeRenderRequest``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ComposeContainer(StrEnum):
    MP4 = "mp4"
    MATROSKA = "mkv"


class ComposeVideoCodec(StrEnum):
    H264 = "libx264"
    FFV1 = "ffv1"


class ComposeAudioCodec(StrEnum):
    AAC = "aac"
    FLAC = "flac"


class ComposePixelFormat(StrEnum):
    YUV420P = "yuv420p"
    YUV444P16LE = "yuv444p16le"


class ComposeAudioSampleFormat(StrEnum):
    S32 = "s32"


@dataclass(frozen=True)
class ComposeRenderPolicy:
    """A closed, typed set of settings needed throughout compose rendering."""

    container: ComposeContainer
    video_codec: ComposeVideoCodec
    audio_codec: ComposeAudioCodec
    pixel_format: ComposePixelFormat
    audio_sample_format: ComposeAudioSampleFormat | None = None

    @property
    def suffix(self) -> str:
        return f".{self.container.value}"


DEFAULT_COMPOSE_RENDER_POLICY = ComposeRenderPolicy(
    container=ComposeContainer.MP4,
    video_codec=ComposeVideoCodec.H264,
    audio_codec=ComposeAudioCodec.AAC,
    pixel_format=ComposePixelFormat.YUV420P,
)

LOSSLESS_COMPOSE_RENDER_POLICY = ComposeRenderPolicy(
    container=ComposeContainer.MATROSKA,
    video_codec=ComposeVideoCodec.FFV1,
    audio_codec=ComposeAudioCodec.FLAC,
    pixel_format=ComposePixelFormat.YUV444P16LE,
    audio_sample_format=ComposeAudioSampleFormat.S32,
)
