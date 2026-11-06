"""Centralized, code-defined capability registry data.

Adding a capability = add a CapabilityDef here (and, if it should do real work,
register a handler in app/capabilities/registry.py).
"""

from __future__ import annotations

from app.capabilities.compose_contract import (
    COMPOSE_DEFAULT_AUDIO_MODE,
    COMPOSE_DEFAULT_MUSIC_VOLUME,
    ComposeAudioMode,
)
from app.capabilities.media_options import (
    AUDIO_CODEC_DESCRIPTIONS,
    AUDIO_CONTAINER_DESCRIPTIONS,
    AUDIO_TRANSCODE_OPTIONS,
    VIDEO_CODEC_DESCRIPTIONS,
    VIDEO_QUALITY_CRF,
    VIDEO_QUALITY_DESCRIPTIONS,
    VIDEO_TRANSCODE_OPTIONS,
)
from app.capabilities.user_text import apply_user_text
from app.composition.export_presets import project_export_input_schema
from app.models.enums import AiContentEffect, CostClass, MediaType
from app.schemas.capability import (
    CapabilityDefinition,
    CapabilityLayer,
    DeploymentRequirement,
    ImplementationReadiness,
    PrimaryAssetApplicability,
    ProjectContextApplicability,
)

# Definitions hold code-owned readiness; API reads add deployment availability.
CapabilityDef = CapabilityDefinition

def _string(title: str, description: str, **keywords: object) -> dict:
    return {"type": "string", "title": title, "description": description, **keywords}


def _number(title: str, description: str, **keywords: object) -> dict:
    return {"type": "number", "title": title, "description": description, **keywords}


def _integer(title: str, description: str, **keywords: object) -> dict:
    return {"type": "integer", "title": title, "description": description, **keywords}


def _asset_id(title: str, description: str, *, role: str, media_types: list[str]) -> dict:
    return _string(
        title,
        description,
        format="uuid",
        **{"x-media-asset": {"role": role, "media_types": media_types}},
    )


def _asset_ids(
    title: str,
    description: str,
    *,
    role: str,
    media_types: list[str],
    min_items: int,
) -> dict:
    item_schema = _asset_id(title, description, role=role, media_types=media_types)
    return {
        "type": "array",
        "title": title,
        "description": description,
        "items": item_schema,
        "minItems": min_items,
        # Preserve the legacy container annotation from the same declaration.
        "x-media-asset": item_schema["x-media-asset"].copy(),
    }


_ASSET_INPUT = {
    "type": "object",
    "properties": {
        "asset_id": _string(
            "Input asset", "Asset processed by this capability.", format="uuid"
        )
    },
}


def _def(
    *,
    id: str,
    title: str,
    description: str,
    media: list[MediaType],
    cost_class: CostClass,
    implementation: ImplementationReadiness,
    layer: CapabilityLayer,
    tool: str,
    output_schema: dict,
    input_schema: dict | None = None,
    enabled: bool = True,
    deployment_requirement: DeploymentRequirement = DeploymentRequirement.LOCAL,
    primary_asset: PrimaryAssetApplicability = PrimaryAssetApplicability.REQUIRED,
    project_context: ProjectContextApplicability = ProjectContextApplicability.DERIVED,
    ai_content_effect: AiContentEffect = AiContentEffect.none,
    client_materialization_media_types: list[MediaType] | None = None,
) -> CapabilityDef:
    return CapabilityDef(
        id=id,
        title=title,
        description=description,
        layer=layer,
        tool=tool,
        input_schema=input_schema or _ASSET_INPUT,
        output_schema=output_schema,
        supported_media_types=media,
        cost_class=cost_class,
        enabled=enabled,
        implementation=implementation,
        ai_content_effect=ai_content_effect,
        client_materialization_media_types=client_materialization_media_types or [],
        deployment_requirement=deployment_requirement,
        applicability={
            "primary_asset": primary_asset,
            "project_context": project_context,
        },
    )


# Output schema shared by all file-producing deterministic capabilities.
_FILE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "outputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "format": "uuid"},
                    "filename": {"type": "string"},
                    "media_type": {"type": "string"},
                    "size_bytes": {"type": "integer"},
                    "download_path": {"type": "string"},
                },
            },
        },
        "size_comparison": {
            "type": "object",
            "description": (
                "Present when the Job has a primary input Asset and every output has "
                "its media type. Sizes in bytes; outputs are summed; ratio is "
                "output/input, so a value above 1 means the result is larger."
            ),
            "properties": {
                "input_size_bytes": {"type": "integer"},
                "output_size_bytes": {"type": "integer"},
                "delta_bytes": {"type": "integer"},
                "ratio": {"type": "number"},
            },
            "required": ["input_size_bytes", "output_size_bytes", "delta_bytes", "ratio"],
        },
    },
}

_NUM = {"type": "number"}
_INT = {"type": "integer"}
_V, _A, _I = [MediaType.video], [MediaType.audio], [MediaType.image]
_VA = [MediaType.video, MediaType.audio]

# The matrix in app.capabilities.media_options is also intersected with the
# smaller set verified by project.export's delivery renderer.


def _transcode_codec_schema(
    field: str,
    title: str,
    description: str,
    *,
    options: dict[str, dict[str, list[str]]] = VIDEO_TRANSCODE_OPTIONS,
) -> dict:
    matrix = {
        container: choices[field] for container, choices in options.items()
    }
    codec_descriptions = (
        VIDEO_CODEC_DESCRIPTIONS if field == "video_codec" else AUDIO_CODEC_DESCRIPTIONS
    )
    enum = list(dict.fromkeys(codec for codecs in matrix.values() for codec in codecs))
    return _string(
        title,
        description,
        enum=enum,
        **{
            "x-enum-descriptions": {
                codec: codec_descriptions[codec] for codec in enum
            },
            "x-option-constraints": {
                "depends_on": "format",
                "allowed_values": matrix,
            },
        },
    )


_VIDEO_TRANSCODE_SCHEMA_CONSTRAINTS = [
    {
        "if": {
            "properties": {"format": {"const": container}},
            "required": ["format"],
        },
        "then": {
            "properties": {
                field: {"enum": options[field]}
                for field in ("video_codec", "audio_codec")
            }
        },
    }
    for container, options in VIDEO_TRANSCODE_OPTIONS.items()
]

_AUDIO_TRANSCODE_SCHEMA_CONSTRAINTS = [
    {
        "if": {
            "properties": {"format": {"const": container}},
            "required": ["format"],
        },
        "then": {"properties": {"codec": {"enum": options["codec"]}}},
    }
    for container, options in AUDIO_TRANSCODE_OPTIONS.items()
]

# Single-asset delivery presets (D-08). Same named destinations and encoding
# settings as project.export, without its Advanced escape hatch.
VIDEO_EXPORT_PRESETS: dict[str, str] = {
    "web": "MP4/H.264 for web playback; fits within 1920x1080 without upscaling.",
    "social": "MP4/H.264 padded to the selected square or portrait canvas.",
    "preview": "Lightweight MP4/H.264 fitting within 1280x720 without upscaling.",
    "archival": "Lossless MKV using FFV1 video and FLAC audio; usually much larger "
                "than the source.",
}

# Declarative specs for the deterministic local-tool capabilities.
_DETERMINISTIC_SPECS: list[dict] = [
    {
        "id": "video.trim",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Video Trim",
        "description": (
            "Keep one specified interval from a video and return one video. "
            "Use this to keep from a time onward (omit the end) or keep a time range; "
            "use video.split to divide the complete input into multiple segments."
        ),
        "media": _V,
        "properties": {
            "start": _number(
                "Start time",
                "Start of the interval to keep, in seconds. Must be before the video ends.",
                minimum=0,
                **{"x-unit": "seconds"},
            ),
            "end": _number(
                "End time",
                "Optional end of the interval to keep, in seconds; must be after the start. "
                "Omit it, or give a time past the end of the video, to keep until the end.",
                exclusiveMinimum=0,
                **{"x-unit": "seconds"},
            ),
        },
        "required": ["start"],
    },
    {
        "id": "video.split",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Video Split",
        "description": (
            "Divide the complete input video at the given time markers and return "
            "multiple segment Assets. Cuts are frame-resolution accurate, not keyframe-only; "
            "the first source soundtrack is retained when present. "
            "Splitting does not choose which segment to keep; "
            "use video.trim to keep one interval."
        ),
        "media": _V,
        "properties": {
            "markers": {
                "type": "array",
                "title": "Split markers",
                "description": (
                    "Strictly increasing, distinct positive times in seconds before the video "
                    "ends. Each interval must contain video frames."
                ),
                "items": {"type": "number", "exclusiveMinimum": 0},
                "minItems": 1,
                "uniqueItems": True,
                "x-unit": "seconds",
            }
        },
        "required": ["markers"],
    },
    {
        "id": "video.concat",
        "layer": CapabilityLayer.COMPOSITION,
        "tool": "FFmpeg",
        "title": "Video Concatenate",
        "description": (
            "Concatenate selected video clips into one video, "
            "preserving their selected order."
        ),
        "media": _V,
        "primary_asset": PrimaryAssetApplicability.OPTIONAL,
        "properties": {
            "asset_ids": _asset_ids(
                "Video clips",
                "Video assets in the exact order they will be concatenated.",
                role="video_segment",
                media_types=["video"],
                min_items=2,
            )
        },
        "required": ["asset_ids"],
    },
    {
        "id": "video.transcode",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Video Transcode",
        "description": (
            "Transcode a video to a supported container and optional codec combination. "
            "Omitting a codec lets FFmpeg choose its default for the selected container. "
            "Optional quality controls compression with an explicit video codec; "
            "high and balanced do not guarantee an output file size, while compressed "
            "also caps the video bitrate at 75% of the source's when it is known."
        ),
        "media": _V,
        "properties": {
            "format": _string(
                "Output container",
                "Container extension for the output video, without a leading dot.",
                enum=list(VIDEO_TRANSCODE_OPTIONS),
                **{
                    "x-enum-descriptions": {
                        "mp4": "MPEG-4 Part 14",
                        "mov": "QuickTime Movie",
                        "mkv": "Matroska",
                        "webm": "WebM",
                    }
                },
            ),
            "video_codec": _transcode_codec_schema(
                "video_codec",
                "Video codec",
                "Optional supported video encoder. Omit to let FFmpeg choose for the container.",
            ),
            "audio_codec": _transcode_codec_schema(
                "audio_codec",
                "Audio codec",
                "Optional supported audio encoder. Omit to let FFmpeg choose for the container.",
            ),
            "quality": _string(
                "Video quality",
                "Optional codec-specific quality/compression level; requires video_codec. "
                "Omit to preserve existing encoder defaults. No exact output file size "
                "is guaranteed; dimensions, frame rate and audio settings are unchanged.",
                enum=list(VIDEO_QUALITY_DESCRIPTIONS),
                **{
                    "x-enum-descriptions": VIDEO_QUALITY_DESCRIPTIONS,
                    "x-option-constraints": {
                        "depends_on": "video_codec",
                        "allowed_values": {
                            codec: list(levels) for codec, levels in VIDEO_QUALITY_CRF.items()
                        },
                    },
                },
            ),
        },
        "required": ["format"],
        "allOf": [
            *_VIDEO_TRANSCODE_SCHEMA_CONSTRAINTS,
            {"if": {"required": ["quality"]}, "then": {"required": ["video_codec"]}},
        ],
    },
    {
        "id": "video.export",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Video Export",
        "description": (
            "Produce a delivery copy of one video for a named destination. Web, Social "
            "and Preview are quality-targeted H.264/AAC MP4 that never upscale; output "
            "size is not capped, so an already efficiently encoded source can come out "
            "larger (reported in size_comparison). Archival is lossless FFV1/FLAC MKV "
            "and is usually much larger. Use video.transcode for explicit "
            "container/codec control and project.export for saved project compositions."
        ),
        "media": _V,
        "properties": {
            "preset": _string(
                "Export preset",
                "Named delivery destination.",
                enum=list(VIDEO_EXPORT_PRESETS),
                **{"x-enum-descriptions": VIDEO_EXPORT_PRESETS},
            ),
            "social_format": _string(
                "Social canvas",
                "Required for the social preset; orientation is never inferred.",
                enum=["square", "portrait"],
                **{
                    "x-enum-descriptions": {
                        "square": "1080 × 1080 square",
                        "portrait": "1080 × 1920 portrait",
                    },
                    "x-option-constraints": {
                        "depends_on": "preset",
                        "allowed_values": {"social": ["square", "portrait"]},
                    },
                },
            ),
        },
        "required": ["preset"],
        "allOf": [
            {
                "if": {"properties": {"preset": {"const": "social"}}, "required": ["preset"]},
                "then": {"required": ["social_format"]},
                "else": {"not": {"required": ["social_format"]}},
            }
        ],
    },
    {
        "id": "video.mute",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Video Mute",
        "description": "Remove the audio track from a video.",
        "media": _V,
        "properties": {},
    },
    {
        "id": "video.crop",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Video Crop",
        "description": (
            "Crop a video to a rectangle (x, y, width, height). Coordinates refer to "
            "the video as displayed, after any rotation, and the rectangle must fit "
            "inside it. Odd sizes or offsets may be rounded down to even numbers."
        ),
        "media": _V,
        "properties": {
            "x": _integer(
                "Left offset", "Distance from the left edge of the video.",
                minimum=0, **{"x-unit": "pixels"},
            ),
            "y": _integer(
                "Top offset", "Distance from the top edge of the video.",
                minimum=0, **{"x-unit": "pixels"},
            ),
            "width": _integer(
                "Width", "Width of the area to keep.", minimum=1, **{"x-unit": "pixels"}
            ),
            "height": _integer(
                "Height", "Height of the area to keep.", minimum=1, **{"x-unit": "pixels"}
            ),
        },
        "required": ["x", "y", "width", "height"],
    },
    {
        "id": "video.resize",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Video Resize",
        "description": (
            "Resize a video as displayed. Give a width, a height, or both; when only one "
            "is given, the other follows the source's aspect ratio. With both, fit "
            "chooses stretch (exact size), contain or cover. Sizes are rounded down to "
            "even numbers, as standard video formats require. The output reports its "
            "size and aspect_ratio_change."
        ),
        "media": _V,
        "properties": {
            "width": _integer(
                "Width",
                "Output width. Omit it to keep the aspect ratio from the height.",
                minimum=2,
                **{"x-unit": "pixels"},
            ),
            "height": _integer(
                "Height",
                "Output height. Omit it to keep the aspect ratio from the width.",
                minimum=2,
                **{"x-unit": "pixels"},
            ),
            "fit": _string(
                "Fit",
                "How to use both sides when width and height are given: stretch to "
                "exactly that size, contain inside it, or cover it and crop the overflow. "
                "Requires both width and height.",
                default="stretch",
                enum=["stretch", "contain", "cover"],
                **{
                    "x-enum-descriptions": {
                        "stretch": "Exactly width x height; the picture is distorted if "
                                   "the aspect ratio differs (reported as "
                                   "aspect_ratio_change).",
                        "contain": "As large as possible inside width x height, keeping the "
                                   "aspect ratio; one side may come out smaller.",
                        "cover": "Fills width x height exactly, keeping the aspect ratio "
                                 "and cropping the overflow from the centre.",
                    }
                },
            ),
        },
        "allOf": [
            {"anyOf": [{"required": ["width"]}, {"required": ["height"]}]},
            {"if": {"required": ["fit"]}, "then": {"required": ["width", "height"]}},
        ],
    },
    {
        "id": "video.thumbnail",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Video Thumbnail",
        "description": "Extract a single frame at a timestamp as an image.",
        "media": _V,
        "properties": {
            "timestamp": _number(
                "Timestamp",
                "Time of the frame to extract, in seconds; must be before the video ends.",
                minimum=0,
                **{"x-unit": "seconds"},
            )
        },
        "required": ["timestamp"],
    },
    {
        "id": "image.resize",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "ImageMagick",
        "title": "Image Resize",
        "description": (
            "Resize the first frame of an image, as displayed after EXIF orientation. "
            "Give a width, a height, or both; when only one is given the other follows "
            "the image's aspect ratio. With both, fit chooses stretch (exactly that "
            "size; the picture may be distorted), contain or cover. Optional quality "
            "applies to JPEG and WebP images. The output reports its size and "
            "aspect_ratio_change."
        ),
        "media": _I,
        "properties": {
            "width": _integer(
                "Width", "Output width. Omit it to keep the aspect ratio from the height.",
                minimum=1, **{"x-unit": "pixels"},
            ),
            "height": _integer(
                "Height", "Output height. Omit it to keep the aspect ratio from the width.",
                minimum=1, **{"x-unit": "pixels"},
            ),
            "fit": _string(
                "Fit",
                "How to use both sides when width and height are given: stretch to "
                "exactly that size, contain inside it, or cover it and crop the overflow. "
                "Requires both width and height.",
                default="stretch",
                enum=["stretch", "contain", "cover"],
                **{
                    "x-enum-descriptions": {
                        "stretch": "Exactly width x height; the picture is distorted if "
                                   "the aspect ratio differs (reported as "
                                   "aspect_ratio_change).",
                        "contain": "As large as possible inside width x height, keeping the "
                                   "aspect ratio; one side may come out smaller.",
                        "cover": "Fills width x height exactly, keeping the aspect ratio "
                                 "and cropping the overflow from the centre.",
                    }
                },
            ),
            "quality": _integer(
                "Quality",
                "Encoding quality for JPEG and WebP images (1-100); higher keeps more "
                "detail. Omit to let ImageMagick follow the source's quality.",
                minimum=1, maximum=100, **{"x-unit": "percent"},
            ),
        },
        "allOf": [
            {"anyOf": [{"required": ["width"]}, {"required": ["height"]}]},
            {"if": {"required": ["fit"]}, "then": {"required": ["width", "height"]}},
        ],
    },
    {
        "id": "image.crop",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "ImageMagick",
        "title": "Image Crop",
        "description": (
            "Crop an image to a rectangle (x, y, width, height). Coordinates refer to "
            "the first frame as displayed, after EXIF orientation, and the rectangle "
            "must fit inside it."
        ),
        "media": _I,
        "properties": {
            "x": _integer(
                "Left offset", "Distance from the left edge of the image.",
                minimum=0, **{"x-unit": "pixels"},
            ),
            "y": _integer(
                "Top offset", "Distance from the top edge of the image.",
                minimum=0, **{"x-unit": "pixels"},
            ),
            "width": _integer(
                "Width", "Width of the area to keep.", minimum=1, **{"x-unit": "pixels"}
            ),
            "height": _integer(
                "Height", "Height of the area to keep.", minimum=1, **{"x-unit": "pixels"}
            ),
        },
        "required": ["x", "y", "width", "height"],
    },
    {
        "id": "image.format",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "ImageMagick",
        "title": "Image Format Convert",
        "description": (
            "Convert the first frame of an image, as displayed after EXIF orientation, "
            "to JPEG, PNG, WebP or TIFF. JPEG and WebP are lossy: quality (1-100) "
            "defaults to 92 for JPEG and 90 for WebP, and WebP can instead be lossless. "
            "PNG and TIFF are always lossless (TIFF uses LZW). JPEG cannot hold "
            "transparency and is flattened onto white; other formats keep it unless "
            "flatten_transparency is set."
        ),
        "media": _I,
        "properties": {
            "format": _string(
                "Output format",
                "Image format produced by the conversion.",
                enum=["jpeg", "png", "webp", "tiff"],
                **{
                    "x-enum-descriptions": {
                        "jpeg": "JPEG: small lossy files; no transparency.",
                        "png": "PNG: lossless, keeps transparency.",
                        "webp": "WebP: small files for the web; lossy or lossless.",
                        "tiff": "TIFF: lossless (LZW), suited to archiving.",
                    }
                },
            ),
            "quality": _integer(
                "Quality",
                "Lossy encoding quality for JPEG and WebP; higher keeps more detail and "
                "makes larger files. Omit for 92 (JPEG) or 90 (WebP).",
                minimum=1,
                maximum=100,
                **{"x-unit": "percent"},
            ),
            "lossless": {
                "type": "boolean",
                "title": "Lossless WebP",
                "description": "Encode WebP without any loss; cannot be combined with quality.",
                "default": False,
                "x-option-constraints": {
                    "depends_on": "format",
                    "allowed_values": {"webp": [False, True]},
                },
            },
            "flatten_transparency": {
                "type": "boolean",
                "title": "Remove transparency",
                "description": (
                    "Fill transparent areas with white. JPEG output is always flattened."
                ),
                "default": False,
            },
        },
        "required": ["format"],
        "allOf": [
            {
                "if": {"properties": {"format": {"enum": ["png", "tiff"]}}},
                "then": {"not": {"required": ["quality"]}},
            },
            {
                "if": {"properties": {"format": {"not": {"const": "webp"}}}},
                "then": {"properties": {"lossless": {"const": False}}},
            },
            {
                "if": {"properties": {"lossless": {"const": True}}, "required": ["lossless"]},
                "then": {"not": {"required": ["quality"]}},
            },
        ],
    },
    {
        "id": "image.colour.adjust",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "ImageMagick",
        "title": "Image Colour Adjust",
        "description": "Adjust brightness, contrast and saturation of an image.",
        "media": _I,
        "properties": {
            "brightness": _integer(
                "Brightness",
                "Brightness percentage; 0 produces black, 100 leaves it unchanged, "
                "and higher values brighten the image. Product range: 0–200.",
                default=100,
                minimum=0,
                maximum=200,
                **{"x-unit": "percent"},
            ),
            "contrast": _integer(
                "Contrast",
                "Contrast percentage adjustment; negative values reduce contrast, "
                "positive values increase it, and 0 leaves it unchanged.",
                default=0,
                minimum=-100,
                maximum=100,
                **{"x-unit": "percent"},
            ),
            "saturation": _integer(
                "Saturation",
                "Saturation percentage; 0 produces grayscale, 100 leaves it unchanged, "
                "and higher values increase colour intensity. Product range: 0–200.",
                default=100,
                minimum=0,
                maximum=200,
                **{"x-unit": "percent"},
            ),
        },
    },
    {
        "id": "image.upscale",
        "layer": CapabilityLayer.AI_ASSISTED,
        "tool": "Real-ESRGAN (CUDA)",
        "ai_content_effect": AiContentEffect.modified,
        "deployment_requirement": DeploymentRequirement.CUDA_WORKER,
        "title": "Image Upscale",
        "description": "Increase image resolution while preserving detail (Real-ESRGAN).",
        "media": _I,
        "properties": {
            "scale": _integer(
                "Scale",
                "Real-ESRGAN output scale factor.",
                default=4,
                enum=[2, 3, 4],
                **{"x-unit": "ratio"},
            )
        },
    },
    {
        "id": "audio.trim",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Audio Trim",
        "description": (
            "Keep one interval of an audio clip and return one audio clip. Omit the "
            "end to keep from the start time onward."
        ),
        "media": _A,
        "properties": {
            "start": _number(
                "Start time",
                "Start of the interval to keep, in seconds. Must be before the audio ends.",
                minimum=0,
                **{"x-unit": "seconds"},
            ),
            "end": _number(
                "End time",
                "Optional end of the interval to keep, in seconds; must be after the start. "
                "Omit it, or give a time past the end of the audio, to keep until the end.",
                exclusiveMinimum=0,
                **{"x-unit": "seconds"},
            ),
        },
        "required": ["start"],
    },
    {
        "id": "audio.concat",
        "layer": CapabilityLayer.COMPOSITION,
        "tool": "FFmpeg",
        "title": "Audio Concatenate",
        "description": (
            "Concatenate selected audio clips into one audio track, "
            "preserving their selected order."
        ),
        "media": _A,
        "primary_asset": PrimaryAssetApplicability.OPTIONAL,
        "properties": {
            "asset_ids": _asset_ids(
                "Audio clips",
                "Audio assets in the exact order they will be concatenated.",
                role="audio_segment",
                media_types=["audio"],
                min_items=2,
            )
        },
        "required": ["asset_ids"],
    },
    {
        "id": "audio.gain",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Audio Gain",
        "description": (
            "Adjust the soundtrack gain (gain_db). Audio input produces audio; "
            "video input preserves the video stream and produces video with processed audio. "
            "The sound keeps its source sample rate, channels, codec and bitrate."
        ),
        "media": _VA,
        "properties": {
            "gain_db": _number(
                "Gain", "Gain adjustment applied to the audio.", **{"x-unit": "decibels"}
            )
        },
        "required": ["gain_db"],
    },
    {
        "id": "audio.normalize",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Audio Normalize",
        "description": (
            "Normalize soundtrack loudness to a target integrated level (LUFS). Audio input "
            "produces audio; video input preserves the video stream and produces video with "
            "normalized audio. The sound keeps its source sample rate, channels, codec and "
            "bitrate."
        ),
        "media": _VA,
        "properties": {
            "target_i": _number(
                "Target loudness",
                "Integrated loudness target used by FFmpeg loudnorm.",
                default=-16,
                minimum=-70,
                maximum=-5,
                **{"x-unit": "LUFS"},
            )
        },
    },
    {
        "id": "audio.fade",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Audio Fade",
        "description": (
            "Apply fade-in and/or fade-out (seconds) to a soundtrack. Audio input produces "
            "audio; video input preserves the video stream and produces video with faded audio. "
            "The sound keeps its source sample rate, channels, codec and bitrate."
        ),
        "media": _VA,
        "properties": {
            "fade_in": _number(
                "Fade in",
                "Fade-in duration; 0 disables it.",
                default=0,
                minimum=0,
                **{"x-unit": "seconds"},
            ),
            "fade_out": _number(
                "Fade out",
                "Fade-out duration; 0 disables it.",
                default=0,
                minimum=0,
                **{"x-unit": "seconds"},
            ),
        },
    },
    {
        "id": "audio.transcode",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Audio Transcode",
        "description": "Transcode audio to a target container/codec.",
        "media": _A,
        "properties": {
            "format": _string(
                "Output format",
                "Supported audio container extension, without a leading dot.",
                enum=list(AUDIO_TRANSCODE_OPTIONS),
                **{"x-enum-descriptions": AUDIO_CONTAINER_DESCRIPTIONS},
            ),
            "codec": _transcode_codec_schema(
                "codec",
                "Audio codec",
                "Optional supported audio encoder. Omit to use the container default: "
                "MP3 for mp3, 16-bit PCM for wav, AAC for m4a, FLAC for flac, "
                "or Vorbis for ogg.",
                options=AUDIO_TRANSCODE_OPTIONS,
            ),
        },
        "required": ["format"],
        "allOf": _AUDIO_TRANSCODE_SCHEMA_CONSTRAINTS,
    },
    {
        "id": "audio.denoise",
        "layer": CapabilityLayer.AI_ASSISTED,
        "tool": "FFmpeg with RNNoise",
        "ai_content_effect": AiContentEffect.modified,
        "title": "Audio Denoise",
        "description": (
            "Reduce soundtrack background noise with RNNoise. Audio input produces audio; "
            "video input preserves the video stream and produces video with denoised audio. "
            "The sound keeps its source sample rate, channels, codec and bitrate."
        ),
        "media": _VA,
        "properties": {
            "strength": _number(
                "Denoise strength",
                "RNNoise mix from -1 to 1; 0 is original and 1 is fully filtered.",
                default=1.0,
                minimum=-1,
                maximum=1,
                **{"x-unit": "ratio"},
            )
        },
    },
    {
        "id": "video.audio.extract",
        "layer": CapabilityLayer.DETERMINISTIC_MEDIA,
        "tool": "FFmpeg",
        "title": "Extract Video Audio",
        "description": (
            "Extract the first audio track from a video as a separate AAC/M4A audio asset. "
            "Unlike audio transforms on video, this does not return video."
        ),
        "media": _V,
        "properties": {},
    },
    # --- Composition capabilities ---
    {
        "id": "image.slideshow",
        "layer": CapabilityLayer.COMPOSITION,
        "tool": "FFmpeg and ImageMagick",
        "title": "Image Slideshow",
        "description": (
            "Build a video slideshow from the selected images in order, showing each "
            "for the specified duration."
        ),
        "media": _I,
        "primary_asset": PrimaryAssetApplicability.OPTIONAL,
        "properties": {
            "asset_ids": _asset_ids(
                "Slideshow images",
                "Image assets in the exact order they will appear.",
                role="slideshow_image",
                media_types=["image"],
                min_items=1,
            ),
            "seconds_per_image": _number(
                "Seconds per image",
                "Display duration of each image.",
                default=4,
                exclusiveMinimum=0,
                **{"x-unit": "seconds"},
            ),
            "width": _integer(
                "Width",
                "Output video width.",
                default=1920,
                minimum=1,
                **{"x-unit": "pixels"},
            ),
            "height": _integer(
                "Height",
                "Output video height.",
                default=1080,
                minimum=1,
                **{"x-unit": "pixels"},
            ),
        },
        "required": ["asset_ids"],
    },
    {
        "id": "image.kenburns",
        "layer": CapabilityLayer.COMPOSITION,
        "tool": "OpenCV and FFmpeg",
        "title": "Ken Burns Pan & Zoom",
        "description": (
            "Apply a saliency-aware pan-and-zoom animation to a still image, "
            "producing a video clip. The pan/zoom target is auto-detected "
            "(face detection, falling back to edge/contrast density) rather "
            "than user-specified."
        ),
        "media": _I,
        "properties": {
            "duration": _number(
                "Duration",
                "Output clip duration.",
                default=5,
                exclusiveMinimum=0,
                **{"x-unit": "seconds"},
            ),
            "zoom": _number(
                "Zoom",
                "Final zoom ratio; must be greater than 1.",
                default=1.15,
                exclusiveMinimum=1,
                **{"x-unit": "ratio"},
            ),
            "direction": _string(
                "Direction",
                "Whether the animation zooms in or out.",
                default="in",
                enum=["in", "out"],
            ),
            "width": _integer(
                "Width",
                "Output video width.",
                default=1920,
                minimum=1,
                **{"x-unit": "pixels"},
            ),
            "height": _integer(
                "Height",
                "Output video height.",
                default=1080,
                minimum=1,
                **{"x-unit": "pixels"},
            ),
        },
    },
    {
        "id": "media.titlecard",
        "layer": CapabilityLayer.COMPOSITION,
        "tool": "ImageMagick and FFmpeg",
        "title": "Title Card",
        "description": (
            "Generate a title-card video clip from text (no input asset needed). "
            "Use as an intro/section card to feed into video.compose."
        ),
        "media": [],
        "primary_asset": PrimaryAssetApplicability.FORBIDDEN,
        "project_context": ProjectContextApplicability.REQUIRED,
        "properties": {
            "text": _string("Text", "Text rendered on the title card."),
            "duration": _number(
                "Duration",
                "Title-card duration.",
                default=5,
                exclusiveMinimum=0,
                **{"x-unit": "seconds"},
            ),
            "width": _integer(
                "Width",
                "Output video width.",
                default=1920,
                minimum=1,
                **{"x-unit": "pixels"},
            ),
            "height": _integer(
                "Height",
                "Output video height.",
                default=1080,
                minimum=1,
                **{"x-unit": "pixels"},
            ),
            "background": _string(
                "Background colour",
                "ImageMagick colour used for the background.",
                default="#101418",
            ),
            "foreground": _string(
                "Text colour", "ImageMagick colour used for the text.", default="#ffffff"
            ),
        },
        "required": ["text"],
    },
    {
        "id": "video.lowerthird",
        "layer": CapabilityLayer.COMPOSITION,
        "tool": "ImageMagick and FFmpeg",
        "title": "Lower Third",
        "description": (
            "Overlay a lower-third text graphic (name/caption) onto a video "
            "for a given time window."
        ),
        "media": _V,
        "properties": {
            "text": _string("Name", "Primary lower-third text."),
            "caption": _string("Caption", "Optional secondary lower-third text."),
            "start": _number(
                "Start time",
                "Time at which the overlay appears.",
                default=0,
                minimum=0,
                **{"x-unit": "seconds"},
            ),
            "duration": _number(
                "Duration",
                "How long the overlay remains visible.",
                default=5,
                exclusiveMinimum=0,
                **{"x-unit": "seconds"},
            ),
            "position": {
                "type": "string",
                "title": "Position",
                "description": "Horizontal placement along the bottom edge.",
                "enum": ["bottom-left", "bottom-center", "bottom-right"],
                "default": "bottom-left",
            },
            "background": _string(
                "Background colour",
                "ImageMagick colour used for the label background.",
                default="#101418",
            ),
            "foreground": _string(
                "Text colour", "ImageMagick colour used for the text.", default="#ffffff"
            ),
        },
        "required": ["text"],
    },
    {
        "id": "video.subtitle.embed",
        "layer": CapabilityLayer.COMPOSITION,
        "tool": "FFmpeg",
        "title": "Embed Subtitles",
        "description": (
            "Add a subtitle file to a video. Use soft mode for a selectable subtitle "
            "track or burn mode to render subtitles into the picture."
        ),
        "media": _V,
        "properties": {
            "subtitle_asset_id": _asset_id(
                "Subtitle asset",
                "Subtitle file to mux or render into the video.",
                role="subtitle",
                media_types=["subtitle"],
            ),
            "mode": {
                "type": "string",
                "title": "Subtitle mode",
                "description": (
                    "Choose a selectable subtitle track or render text into the picture."
                ),
                "enum": ["soft", "burn"],
                "default": "soft",
                "x-enum-descriptions": {
                    "soft": "Mux a selectable subtitle track without burning it into frames.",
                    "burn": "Render subtitles permanently into the video picture.",
                },
            },
        },
        "required": ["subtitle_asset_id"],
    },
    {
        "id": "audio.mix",
        "layer": CapabilityLayer.COMPOSITION,
        "tool": "FFmpeg",
        "title": "Mix Background Audio",
        "description": (
            "Mix the selected background music or narration track with the primary "
            "video or audio. Choose mix, duck, or replace mode."
        ),
        "media": _VA,
        "properties": {
            "music_asset_id": _asset_id(
                "Background audio",
                "Audio asset used as music or narration.",
                role="background_audio",
                media_types=["audio"],
            ),
            "music_volume": _number(
                "Music volume",
                "Linear volume multiplier for the supplied audio; duck caps it at 0.2.",
                default=0.3,
                **{"x-unit": "ratio"},
            ),
            "mode": {
                "type": "string",
                "title": "Audio mode",
                "description": "How supplied audio combines with the primary asset's audio.",
                "enum": ["mix", "duck", "replace"],
                "default": "mix",
                "x-enum-descriptions": {
                    "mix": "Mix supplied audio with the primary asset's existing audio.",
                    "duck": "Mix supplied audio at no more than a 0.2 volume multiplier.",
                    "replace": "Replace the primary asset's audio with the supplied audio.",
                },
            },
        },
        "required": ["music_asset_id"],
    },
    {
        "id": "video.compose",
        "layer": CapabilityLayer.COMPOSITION,
        "tool": "Internal composition renderer (FFmpeg)",
        "title": "Compose / Export Timeline",
        "description": (
            "Render a final video from ordered video segments normalized to the chosen "
            "dimensions, with an optional audio bed and burned subtitles."
        ),
        "media": _V,
        "primary_asset": PrimaryAssetApplicability.OPTIONAL,
        "properties": {
            "asset_ids": _asset_ids(
                "Video segments",
                "Video assets in the exact timeline order.",
                role="video_segment",
                media_types=["video"],
                min_items=1,
            ),
            "width": _integer(
                "Width",
                "Composition output width.",
                default=1920,
                minimum=1,
                **{"x-unit": "pixels"},
            ),
            "height": _integer(
                "Height",
                "Composition output height.",
                default=1080,
                minimum=1,
                **{"x-unit": "pixels"},
            ),
            "audio_asset_id": _asset_id(
                "Audio bed",
                "Optional audio asset mixed into or substituted for timeline audio.",
                role="background_audio",
                media_types=["audio"],
            ),
            "audio_mode": {
                "type": "string",
                "title": "Audio mode",
                "description": "How the optional audio bed combines with timeline audio.",
                "enum": [mode.value for mode in ComposeAudioMode],
                "default": COMPOSE_DEFAULT_AUDIO_MODE.value,
                "x-enum-descriptions": {
                    "mix": "Mix the audio bed with timeline audio.",
                    "duck": "Mix the audio bed at no more than a 0.2 volume multiplier.",
                    "replace": "Replace timeline audio with the audio bed.",
                },
            },
            "music_volume": _number(
                "Music volume",
                "Linear volume multiplier for the optional audio bed.",
                default=COMPOSE_DEFAULT_MUSIC_VOLUME,
                **{"x-unit": "ratio"},
            ),
            "subtitle_asset_id": _asset_id(
                "Subtitle asset",
                "Optional subtitle file burned into the final composition.",
                role="subtitle",
                media_types=["subtitle"],
            ),
        },
        "required": ["asset_ids"],
    },
]


CAPABILITIES: list[CapabilityDef] = [
    _def(
        id="audio.transcribe",
        layer=CapabilityLayer.AI_ASSISTED,
        tool="Configured inference backend",
        ai_content_effect=AiContentEffect.generated,
        client_materialization_media_types=[MediaType.subtitle],
        implementation=ImplementationReadiness.AVAILABLE,
        deployment_requirement=DeploymentRequirement.INFERENCE_PROVIDER,
        title="Audio Transcription",
        description="Transcribe speech in an audio asset to text.",
        media=[MediaType.audio, MediaType.video],
        cost_class=CostClass.hosted_ai,
        output_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}, "language": {"type": "string"}},
            "required": ["text"],
        },
    ),
    _def(
        id="subtitle.autogenerate",
        layer=CapabilityLayer.AI_ASSISTED,
        tool="Configured inference backend",
        ai_content_effect=AiContentEffect.generated,
        implementation=ImplementationReadiness.AVAILABLE,
        deployment_requirement=DeploymentRequirement.INFERENCE_PROVIDER,
        title="Subtitle Auto-generation",
        description=(
            "Generate time-coded SRT subtitles from the speech in an audio or video "
            "asset and return them as a new subtitle Asset, ready for "
            "video.subtitle.embed. The first soundtrack is transcribed with segment "
            "timestamps by the configured inference backend."
        ),
        media=[MediaType.audio, MediaType.video],
        cost_class=CostClass.hosted_ai,
        input_schema={
            "type": "object",
            "properties": {
                "asset_id": _string(
                    "Input asset", "Audio or video whose speech is subtitled.", format="uuid"
                ),
                "language": _string(
                    "Spoken language",
                    "Optional ISO-639-1 code of the spoken language (e.g. en, el); "
                    "omit to detect it automatically.",
                    pattern="^[a-z]{2}$",
                ),
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "outputs": _FILE_OUTPUT_SCHEMA["properties"]["outputs"],  # type: ignore[index]
                "language": {"type": ["string", "null"]},
                "segment_count": {"type": "integer", "minimum": 1},
            },
            "required": ["segment_count"],
        },
    ),
    _def(
        id="audio.tts",
        layer=CapabilityLayer.AI_ASSISTED,
        tool="Configured inference backend",
        ai_content_effect=AiContentEffect.generated,
        implementation=ImplementationReadiness.AVAILABLE,
        deployment_requirement=DeploymentRequirement.INFERENCE_PROVIDER,
        title="Text to Speech",
        description="Synthesize speech audio from text.",
        media=[],
        cost_class=CostClass.hosted_ai,
        primary_asset=PrimaryAssetApplicability.FORBIDDEN,
        project_context=ProjectContextApplicability.REQUIRED,
        input_schema={
            "type": "object",
            "properties": {
                "text": _string("Text", "Text to synthesize as speech."),
                "voice": _string("Voice", "Optional provider voice identifier."),
            },
            "required": ["text"],
        },
        output_schema={"type": "object", "properties": {"audio_url": {"type": "string"}}},
    ),
    _def(
        id="audio.music.generate",
        layer=CapabilityLayer.AI_ASSISTED,
        tool="Not implemented",
        ai_content_effect=AiContentEffect.generated,
        implementation=ImplementationReadiness.UNAVAILABLE,
        title="Music Generation",
        description="Generate music audio from a text prompt.",
        media=[],
        cost_class=CostClass.hosted_ai,
        primary_asset=PrimaryAssetApplicability.FORBIDDEN,
        project_context=ProjectContextApplicability.REQUIRED,
        input_schema={
            "type": "object",
            "properties": {"prompt": _string("Prompt", "Description of the music to generate.")},
            "required": ["prompt"],
        },
        output_schema={"type": "object", "properties": {"audio_url": {"type": "string"}}},
    ),
    _def(
        id="audio.separate.stems",
        layer=CapabilityLayer.AI_ASSISTED,
        tool="Not implemented",
        ai_content_effect=AiContentEffect.modified,
        implementation=ImplementationReadiness.UNAVAILABLE,
        title="Stem Separation",
        description="Separate an audio track into instrument/vocal stems.",
        media=[MediaType.audio],
        cost_class=CostClass.hosted_ai,
        output_schema={
            "type": "object",
            "properties": {"stems": {"type": "array", "items": {"type": "string"}}},
        },
    ),
    _def(
        id="image.background.remove",
        layer=CapabilityLayer.AI_ASSISTED,
        tool="Not implemented",
        ai_content_effect=AiContentEffect.modified,
        implementation=ImplementationReadiness.UNAVAILABLE,
        title="Background Removal",
        description="Remove the background from an image.",
        media=[MediaType.image],
        cost_class=CostClass.hosted_ai,
        output_schema={"type": "object", "properties": {"image_url": {"type": "string"}}},
    ),
    _def(
        id="image.inpaint",
        layer=CapabilityLayer.AI_ASSISTED,
        tool="Not implemented",
        ai_content_effect=AiContentEffect.modified,
        implementation=ImplementationReadiness.UNAVAILABLE,
        title="Image Inpainting",
        description="Fill or replace masked regions of an image.",
        media=[MediaType.image],
        cost_class=CostClass.hosted_ai,
        input_schema={
            "type": "object",
            "properties": {
                "asset_id": _asset_id(
                    "Source image",
                    "Image asset containing the region to inpaint.",
                    role="source_image",
                    media_types=["image"],
                ),
                "mask_url": _string(
                    "Mask URL", "URL of a mask identifying the image regions to fill or replace."
                ),
                "prompt": _string(
                    "Replacement description",
                    "Describe the content you want in the masked image regions.",
                ),
            },
        },
        output_schema={"type": "object", "properties": {"image_url": {"type": "string"}}},
    ),
    _def(
        id="image.restore.face",
        layer=CapabilityLayer.AI_ASSISTED,
        tool="Not implemented",
        ai_content_effect=AiContentEffect.modified,
        implementation=ImplementationReadiness.UNAVAILABLE,
        title="Face Restoration",
        description="Restore and enhance faces in an image.",
        media=[MediaType.image],
        cost_class=CostClass.hosted_ai,
        output_schema={"type": "object", "properties": {"image_url": {"type": "string"}}},
    ),
    _def(
        id="image.caption",
        layer=CapabilityLayer.AI_ASSISTED,
        tool="Configured inference backend",
        ai_content_effect=AiContentEffect.generated,
        implementation=ImplementationReadiness.AVAILABLE,
        deployment_requirement=DeploymentRequirement.INFERENCE_PROVIDER,
        title="Image Captioning",
        description="Generate a natural-language caption for an image.",
        media=[MediaType.image],
        cost_class=CostClass.hosted_ai,
        output_schema={
            "type": "object",
            "properties": {"caption": {"type": "string", "minLength": 1, "pattern": r"\S"}},
            "required": ["caption"],
        },
    ),
    _def(
        id="image.tag",
        layer=CapabilityLayer.AI_ASSISTED,
        tool="Not implemented",
        ai_content_effect=AiContentEffect.generated,
        implementation=ImplementationReadiness.UNAVAILABLE,
        title="Image Tagging",
        description="Generate descriptive tags/keywords for an image.",
        media=[MediaType.image],
        cost_class=CostClass.hosted_ai,
        output_schema={
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        },
    ),
    _def(
        id="video.upscale",
        layer=CapabilityLayer.AI_ASSISTED,
        tool="Not implemented",
        ai_content_effect=AiContentEffect.modified,
        implementation=ImplementationReadiness.UNAVAILABLE,
        title="Video Upscale",
        description="Increase video resolution.",
        media=[MediaType.video],
        cost_class=CostClass.future_gpu,
        output_schema={"type": "object", "properties": {"video_url": {"type": "string"}}},
    ),
    # --- Deterministic local-tool capabilities (FFmpeg / ImageMagick / Real-ESRGAN) ---
    # All produce one or more output files, registered as derived Assets and
    # listed under output.outputs[{asset_id, filename, media_type, ...}].
    *(
        _def(
            id=spec["id"],
            layer=spec["layer"],
            tool=spec["tool"],
            ai_content_effect=spec.get("ai_content_effect", AiContentEffect.none),
            implementation=ImplementationReadiness.AVAILABLE,
            deployment_requirement=spec.get(
                "deployment_requirement", DeploymentRequirement.LOCAL
            ),
            title=spec["title"],
            description=spec["description"],
            media=spec["media"],
            cost_class=CostClass.deterministic,
            input_schema={
                "type": "object",
                "properties": spec["properties"],
                "required": spec.get("required", []),
                **({"allOf": spec["allOf"]} if "allOf" in spec else {}),
            },
            output_schema=_FILE_OUTPUT_SCHEMA,
            primary_asset=spec.get(
                "primary_asset", PrimaryAssetApplicability.REQUIRED
            ),
            project_context=spec.get(
                "project_context", ProjectContextApplicability.DERIVED
            ),
        )
        for spec in _DETERMINISTIC_SPECS
    ),
    # Deterministic, but JSON-only (no output file) so it doesn't fit
    # _FILE_OUTPUT_SCHEMA / _DETERMINISTIC_SPECS above.
    _def(
        id="video.shot.detect",
        layer=CapabilityLayer.DETERMINISTIC_MEDIA,
        tool="PySceneDetect",
        implementation=ImplementationReadiness.AVAILABLE,
        title="Shot Detection",
        description="Detect shot/scene boundaries in a video (PySceneDetect content-aware cuts).",
        media=_V,
        cost_class=CostClass.deterministic,
        input_schema={
            "type": "object",
            "properties": {
                "threshold": _number(
                    "Detection threshold",
                    "Content-change threshold on a 0–255 score. Lower values are more "
                    "sensitive and can detect more cuts, including small changes; higher "
                    "values detect fewer cuts and may miss subtle scene changes.",
                    default=27.0,
                    minimum=0,
                    maximum=255,
                )
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "shots": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": _INT,
                            "start": _NUM,
                            "end": _NUM,
                            "start_frame": _INT,
                            "end_frame": _INT,
                        },
                    },
                }
            },
        },
    ),
    _def(
        id="project.export",
        layer=CapabilityLayer.COMPOSITION,
        tool="Internal composition renderer (FFmpeg)",
        implementation=ImplementationReadiness.AVAILABLE,
        title="Export Project Composition",
        description=(
            "Render the selected immutable project composition revision using a named "
            "delivery preset. Social requires an explicit canvas; Archival is locked "
            "lossless FFV1/FLAC; Advanced accepts only constrained H.264/AAC settings."
        ),
        media=[],
        cost_class=CostClass.deterministic,
        primary_asset=PrimaryAssetApplicability.FORBIDDEN,
        project_context=ProjectContextApplicability.REQUIRED,
        input_schema=project_export_input_schema(),
        output_schema=_FILE_OUTPUT_SCHEMA,
    ),
]

CAPABILITIES = apply_user_text(CAPABILITIES)

CAPABILITIES_BY_ID: dict[str, CapabilityDef] = {c.id: c for c in CAPABILITIES}
