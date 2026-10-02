"""Closed typed delivery settings consumed by the project export renderer."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExportContainer(StrEnum):
    MP4 = "mp4"
    MATROSKA = "mkv"


class ExportVideoCodec(StrEnum):
    H264 = "libx264"
    FFV1 = "ffv1"


class ExportAudioCodec(StrEnum):
    AAC = "aac"
    FLAC = "flac"


class ExportPixelFormat(StrEnum):
    YUV420P = "yuv420p"
    YUV444P16LE = "yuv444p16le"


class CompositionPolicyId(StrEnum):
    DEFAULT = "default"
    LOSSLESS = "lossless"


class FitPolicy(StrEnum):
    PRESERVE = "preserve"
    FIT = "fit"
    PAD = "pad"


class EffectiveExportSettings(BaseModel):
    """Fully resolved, serializable settings; no arbitrary FFmpeg arguments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    container: ExportContainer
    video_codec: ExportVideoCodec
    audio_codec: ExportAudioCodec
    composition_policy: CompositionPolicyId
    pixel_format: ExportPixelFormat
    profile: str | None = None
    crf: int | None = Field(default=None, ge=0, le=51)
    max_width: int | None = Field(default=None, ge=2, le=8192)
    max_height: int | None = Field(default=None, ge=2, le=8192)
    target_width: int | None = Field(default=None, ge=2, le=8192)
    target_height: int | None = Field(default=None, ge=2, le=8192)
    fit: FitPolicy
    audio_bitrate_kbps: int | None = Field(default=None, ge=32, le=512)
    audio_channels: int | None = Field(default=None, ge=1, le=8)
    faststart: bool

    @model_validator(mode="after")
    def validate_semantic_combination(self) -> EffectiveExportSettings:
        if (self.max_width is None) != (self.max_height is None):
            raise ValueError("max_width and max_height must be supplied together")
        if (self.target_width is None) != (self.target_height is None):
            raise ValueError("target_width and target_height must be supplied together")
        if self.max_width is not None and self.target_width is not None:
            raise ValueError("max bounds and exact target canvas are mutually exclusive")
        for dimension in (self.max_width, self.max_height, self.target_width, self.target_height):
            if dimension is not None and dimension % 2:
                raise ValueError("H.264 delivery dimensions must be even")

        if self.composition_policy == CompositionPolicyId.LOSSLESS:
            if (
                self.container != ExportContainer.MATROSKA
                or self.video_codec != ExportVideoCodec.FFV1
                or self.audio_codec != ExportAudioCodec.FLAC
                or self.pixel_format != ExportPixelFormat.YUV444P16LE
                or self.profile is not None
                or self.crf is not None
                or self.audio_bitrate_kbps is not None
                or self.audio_channels is not None
                or self.faststart
                or self.max_width is not None
                or self.max_height is not None
                or self.target_width is not None
                or self.target_height is not None
            ):
                raise ValueError("Lossless composition settings are locked")
            return self

        if (
            self.video_codec != ExportVideoCodec.H264
            or self.audio_codec != ExportAudioCodec.AAC
            or self.pixel_format != ExportPixelFormat.YUV420P
            or self.profile != "high"
        ):
            raise ValueError("Delivery export supports only H.264 High, AAC, and yuv420p")
        if self.crf is None or self.audio_bitrate_kbps is None:
            raise ValueError("Delivery exports require CRF and audio bitrate settings")
        if self.audio_channels != 2:
            raise ValueError("Delivery exports use stereo audio")
        if self.faststart != (self.container == ExportContainer.MP4):
            raise ValueError("Faststart is enabled exactly for MP4 delivery")
        if self.fit == FitPolicy.PAD and self.target_width is None:
            raise ValueError("Padding requires an exact target canvas")
        return self


def settings_dict(settings: EffectiveExportSettings) -> dict:
    """Stable JSON-compatible representation stored in immutable Job input."""
    return settings.model_dump(mode="json")
