"""Canonical versioned project-export presets and their public input schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.capabilities.media_options import VIDEO_TRANSCODE_OPTIONS
from app.tools.export_policy import (
    CompositionPolicyId,
    EffectiveExportSettings,
    ExportAudioCodec,
    ExportContainer,
    ExportPixelFormat,
    ExportVideoCodec,
    FitPolicy,
)

EXPORT_PRESET_VERSION = 1


class ExportPresetId(StrEnum):
    WEB = "web"
    SOCIAL = "social"
    PREVIEW = "preview"
    ARCHIVAL = "archival"
    ADVANCED = "advanced"


class SocialFormat(StrEnum):
    SQUARE = "square"
    PORTRAIT = "portrait"


class PresetResolutionKind(StrEnum):
    FIXED = "fixed"
    SOCIAL = "social"
    ADVANCED = "advanced"


@dataclass(frozen=True)
class ExportPresetDefinition:
    id: ExportPresetId
    title: str
    description: str
    resolution_kind: PresetResolutionKind
    settings: EffectiveExportSettings | None = None


_WEB = EffectiveExportSettings(
    container=ExportContainer.MP4,
    video_codec=ExportVideoCodec.H264,
    audio_codec=ExportAudioCodec.AAC,
    composition_policy=CompositionPolicyId.DEFAULT,
    pixel_format=ExportPixelFormat.YUV420P,
    profile="high",
    crf=22,
    max_width=1920,
    max_height=1080,
    fit=FitPolicy.FIT,
    audio_bitrate_kbps=128,
    audio_channels=2,
    faststart=True,
)
_SOCIAL_TEMPLATE = EffectiveExportSettings(
    container=ExportContainer.MP4,
    video_codec=ExportVideoCodec.H264,
    audio_codec=ExportAudioCodec.AAC,
    composition_policy=CompositionPolicyId.DEFAULT,
    pixel_format=ExportPixelFormat.YUV420P,
    profile="high",
    crf=21,
    fit=FitPolicy.PAD,
    audio_bitrate_kbps=128,
    audio_channels=2,
    faststart=True,
    target_width=1080,
    target_height=1080,
)
_PREVIEW = EffectiveExportSettings(
    container=ExportContainer.MP4,
    video_codec=ExportVideoCodec.H264,
    audio_codec=ExportAudioCodec.AAC,
    composition_policy=CompositionPolicyId.DEFAULT,
    pixel_format=ExportPixelFormat.YUV420P,
    profile="high",
    crf=28,
    max_width=1280,
    max_height=720,
    fit=FitPolicy.FIT,
    audio_bitrate_kbps=96,
    audio_channels=2,
    faststart=True,
)
_ARCHIVAL = EffectiveExportSettings(
    container=ExportContainer.MATROSKA,
    video_codec=ExportVideoCodec.FFV1,
    audio_codec=ExportAudioCodec.FLAC,
    composition_policy=CompositionPolicyId.LOSSLESS,
    pixel_format=ExportPixelFormat.YUV444P16LE,
    fit=FitPolicy.PRESERVE,
    faststart=False,
)

EXPORT_PRESETS: dict[ExportPresetId, ExportPresetDefinition] = {
    ExportPresetId.WEB: ExportPresetDefinition(
        ExportPresetId.WEB,
        "Web",
        "MP4/H.264 for web playback; fits within 1920x1080 without upscaling or cropping.",
        PresetResolutionKind.FIXED,
        _WEB,
    ),
    ExportPresetId.SOCIAL: ExportPresetDefinition(
        ExportPresetId.SOCIAL,
        "Social",
        "MP4/H.264 padded to the explicitly selected square or portrait canvas.",
        PresetResolutionKind.SOCIAL,
    ),
    ExportPresetId.PREVIEW: ExportPresetDefinition(
        ExportPresetId.PREVIEW,
        "Preview / Proxy",
        "Lightweight MP4/H.264 fitting within 1280x720 without upscaling.",
        PresetResolutionKind.FIXED,
        _PREVIEW,
    ),
    ExportPresetId.ARCHIVAL: ExportPresetDefinition(
        ExportPresetId.ARCHIVAL,
        "Archival",
        "Locked lossless MKV using FFV1 video and FLAC audio; no overrides.",
        PresetResolutionKind.FIXED,
        _ARCHIVAL,
    ),
    ExportPresetId.ADVANCED: ExportPresetDefinition(
        ExportPresetId.ADVANCED,
        "Advanced",
        "Select a constrained MP4 or MKV H.264/AAC combination and supported quality/size limits.",
        PresetResolutionKind.ADVANCED,
    ),
}

# Advanced is deliberately the intersection of the public video.transcode
# matrix and the combination exercised by this renderer. FFV1/FLAC remains the
# named locked Archival preset rather than an Advanced escape hatch.
_ADVANCED_CONTAINERS = ("mp4", "mkv")
ADVANCED_EXPORT_OPTIONS: dict[str, dict[str, list[str]]] = {
    container: {
        "video_codec": [
            codec
            for codec in VIDEO_TRANSCODE_OPTIONS[container]["video_codec"]
            if codec == ExportVideoCodec.H264.value
        ],
        "audio_codec": [
            codec
            for codec in VIDEO_TRANSCODE_OPTIONS[container]["audio_codec"]
            if codec == ExportAudioCodec.AAC.value
        ],
    }
    for container in _ADVANCED_CONTAINERS
}


def resolve_effective_settings(
    request: dict[str, Any], *, canvas_width: int, canvas_height: int
) -> EffectiveExportSettings:
    """Resolve validated public selections against the canonical registry."""
    try:
        preset_id = ExportPresetId(request["preset"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Unsupported export preset") from exc
    definition = EXPORT_PRESETS[preset_id]
    if definition.resolution_kind == PresetResolutionKind.FIXED:
        assert definition.settings is not None
        return definition.settings
    if definition.resolution_kind == PresetResolutionKind.SOCIAL:
        social_format = SocialFormat(request["social_format"])
        target = (
            (1080, 1080)
            if social_format == SocialFormat.SQUARE
            else (1080, 1920)
        )
        assert _SOCIAL_TEMPLATE is not None
        return _SOCIAL_TEMPLATE.model_copy(
            update={"target_width": target[0], "target_height": target[1]}
        )

    if definition.resolution_kind == PresetResolutionKind.ADVANCED:
        container = ExportContainer(request["format"])
        video_codec = ExportVideoCodec(request["video_codec"])
        audio_codec = ExportAudioCodec(request["audio_codec"])
        options = ADVANCED_EXPORT_OPTIONS.get(container.value)
        if (
            options is None
            or video_codec.value not in options["video_codec"]
            or audio_codec.value not in options["audio_codec"]
        ):
            raise ValueError("Unsupported Advanced container/codec combination")
        width, height = request.get("max_width"), request.get("max_height")
        return EffectiveExportSettings(
            container=container,
            video_codec=video_codec,
            audio_codec=audio_codec,
            composition_policy=CompositionPolicyId.DEFAULT,
            pixel_format=ExportPixelFormat.YUV420P,
            profile="high",
            crf=request.get("crf", 22),
            max_width=width,
            max_height=height,
            fit=FitPolicy.FIT,
            audio_bitrate_kbps=request.get("audio_bitrate_kbps", 128),
            audio_channels=2,
            faststart=container == ExportContainer.MP4,
        )
    raise ValueError("Unsupported export preset configuration")


def project_export_input_schema() -> dict[str, Any]:
    """Canonical frontend/planner contract generated from preset definitions."""
    preset_values = [preset.value for preset in EXPORT_PRESETS]
    codec_values = {
        field: list(
            dict.fromkeys(
                codec
                for options in ADVANCED_EXPORT_OPTIONS.values()
                for codec in options[field]
            )
        )
        for field in ("video_codec", "audio_codec")
    }
    advanced_properties = {
        "format": {
            "type": "string",
            "title": "Container",
            "description": "Advanced export container.",
            "enum": list(ADVANCED_EXPORT_OPTIONS),
            "x-enum-descriptions": {"mp4": "MP4", "mkv": "Matroska"},
        },
        "video_codec": {
            "type": "string",
            "title": "Video codec",
            "description": "Choose a supported Advanced video encoder.",
            "enum": codec_values["video_codec"],
            "x-enum-descriptions": {"libx264": "H.264 / AVC"},
            "x-option-constraints": {
                "depends_on": "format",
                "allowed_values": {
                    container: options["video_codec"]
                    for container, options in ADVANCED_EXPORT_OPTIONS.items()
                },
            },
        },
        "audio_codec": {
            "type": "string",
            "title": "Audio codec",
            "description": "Choose a supported Advanced audio encoder.",
            "enum": codec_values["audio_codec"],
            "x-enum-descriptions": {"aac": "AAC"},
            "x-option-constraints": {
                "depends_on": "format",
                "allowed_values": {
                    container: options["audio_codec"]
                    for container, options in ADVANCED_EXPORT_OPTIONS.items()
                },
            },
        },
        "max_width": {
            "type": "integer",
            "title": "Maximum width",
            "description": "Optional even-pixel output width bound; no upscaling.",
            "minimum": 2,
            "maximum": 8192,
            "multipleOf": 2,
            "x-unit": "pixels",
        },
        "max_height": {
            "type": "integer",
            "title": "Maximum height",
            "description": "Optional even-pixel output height bound; no upscaling.",
            "minimum": 2,
            "maximum": 8192,
            "multipleOf": 2,
            "x-unit": "pixels",
        },
        "crf": {
            "type": "integer",
            "title": "Video quality (CRF)",
            "description": "H.264 constant-rate-factor value; lower values retain more quality.",
            "minimum": 18,
            "maximum": 35,
            "default": 22,
        },
        "audio_bitrate_kbps": {
            "type": "integer",
            "title": "Audio bitrate",
            "description": "AAC audio bitrate in kilobits per second.",
            "minimum": 64,
            "maximum": 320,
            "default": 128,
            "x-unit": "kbps",
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "composition_revision_id": {
                "type": "string",
                "format": "uuid",
                "title": "Composition revision",
                "description": "Exact immutable saved revision to render.",
            },
            "preset": {
                "type": "string",
                "title": "Export preset",
                "description": "Named delivery profile or constrained Advanced settings.",
                "enum": preset_values,
                "x-enum-descriptions": {
                    preset.value: EXPORT_PRESETS[preset].description
                    for preset in EXPORT_PRESETS
                },
            },
            "social_format": {
                "type": "string",
                "title": "Social canvas",
                "description": "Required Social target; orientation is never inferred.",
                "enum": [item.value for item in SocialFormat],
                "x-enum-descriptions": {
                    "square": "1080 × 1080 square",
                    "portrait": "1080 × 1920 portrait",
                },
                "x-option-constraints": {
                    "depends_on": "preset",
                    "allowed_values": {"social": [item.value for item in SocialFormat]},
                },
            },
            **advanced_properties,
        },
        "required": ["composition_revision_id", "preset"],
        "allOf": [
            {
                "if": {
                    "properties": {"preset": {"const": "social"}},
                    "required": ["preset"],
                },
                "then": {"required": ["social_format"]},
                "else": {"not": {"required": ["social_format"]}},
            },
            {
                "if": {
                    "properties": {"preset": {"const": "advanced"}},
                    "required": ["preset"],
                },
                "then": {
                    "required": ["format", "video_codec", "audio_codec"],
                    "dependentRequired": {"max_width": ["max_height"], "max_height": ["max_width"]},
                },
                "else": {
                    "not": {
                        "anyOf": [
                            {"required": ["format"]},
                            {"required": ["video_codec"]},
                            {"required": ["audio_codec"]},
                            {"required": ["max_width"]},
                            {"required": ["max_height"]},
                            {"required": ["crf"]},
                            {"required": ["audio_bitrate_kbps"]},
                        ]
                    }
                },
            },
            *[
                {
                    "if": {
                        "properties": {"format": {"const": container}},
                        "required": ["format"],
                    },
                    "then": {
                        "properties": {
                            "video_codec": {"enum": options["video_codec"]},
                            "audio_codec": {"enum": options["audio_codec"]},
                        }
                    },
                }
                for container, options in ADVANCED_EXPORT_OPTIONS.items()
            ],
        ],
    }
