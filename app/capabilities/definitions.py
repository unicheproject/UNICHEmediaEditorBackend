"""Centralized, code-defined capability registry data.

Adding a capability = add a CapabilityDef here (and, if it should do real work,
register a handler in app/capabilities/registry.py).
"""

from __future__ import annotations

from app.models.enums import CostClass, MediaType
from app.schemas.capability import CapabilityRead

# A CapabilityDef is just a CapabilityRead instance.
CapabilityDef = CapabilityRead

_ASSET_INPUT = {
    "type": "object",
    "properties": {"asset_id": {"type": "string", "format": "uuid"}},
}


def _def(
    *,
    id: str,
    title: str,
    description: str,
    media: list[MediaType],
    cost_class: CostClass,
    output_schema: dict,
    input_schema: dict | None = None,
    enabled: bool = True,
) -> CapabilityDef:
    return CapabilityDef(
        id=id,
        title=title,
        description=description,
        input_schema=input_schema or _ASSET_INPUT,
        output_schema=output_schema,
        supported_media_types=media,
        cost_class=cost_class,
        enabled=enabled,
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
        }
    },
}

_NUM = {"type": "number"}
_INT = {"type": "integer"}
_STR = {"type": "string"}
_IDS = {"type": "array", "items": {"type": "string", "format": "uuid"}}
_UUID = {"type": "string", "format": "uuid"}
_V, _A, _I = [MediaType.video], [MediaType.audio], [MediaType.image]
_VA = [MediaType.video, MediaType.audio]

# Declarative specs for the deterministic local-tool capabilities.
_DETERMINISTIC_SPECS: list[dict] = [
    {
        "id": "video.trim",
        "title": "Video Trim",
        "description": "Trim a video clip to a start/end time range (seconds).",
        "media": _V,
        "properties": {"start": _NUM, "end": _NUM},
        "required": ["start", "end"],
    },
    {
        "id": "video.split",
        "title": "Video Split",
        "description": "Split a video into segments at the given time markers (seconds).",
        "media": _V,
        "properties": {"markers": {"type": "array", "items": _NUM}},
        "required": ["markers"],
    },
    {
        "id": "video.concat",
        "title": "Video Concatenate",
        "description": "Concatenate multiple video clips into one (input.asset_ids).",
        "media": _V,
        "properties": {"asset_ids": _IDS},
        "required": ["asset_ids"],
    },
    {
        "id": "video.transcode",
        "title": "Video Transcode",
        "description": "Transcode a video to a target container/codec.",
        "media": _V,
        "properties": {"format": _STR, "video_codec": _STR, "audio_codec": _STR},
        "required": ["format"],
    },
    {
        "id": "video.mute",
        "title": "Video Mute",
        "description": "Remove the audio track from a video.",
        "media": _V,
        "properties": {},
    },
    {
        "id": "video.crop",
        "title": "Video Crop",
        "description": "Crop a video to a rectangle (x, y, width, height).",
        "media": _V,
        "properties": {"x": _INT, "y": _INT, "width": _INT, "height": _INT},
        "required": ["x", "y", "width", "height"],
    },
    {
        "id": "video.resize",
        "title": "Video Resize",
        "description": "Resize a video to the given dimensions.",
        "media": _V,
        "properties": {"width": _INT, "height": _INT},
        "required": ["width", "height"],
    },
    {
        "id": "video.thumbnail",
        "title": "Video Thumbnail",
        "description": "Extract a single frame at a timestamp as an image.",
        "media": _V,
        "properties": {"timestamp": _NUM},
        "required": ["timestamp"],
    },
    {
        "id": "image.resize",
        "title": "Image Resize",
        "description": "Resize an image to the given dimensions.",
        "media": _I,
        "properties": {"width": _INT, "height": _INT},
        "required": ["width", "height"],
    },
    {
        "id": "image.crop",
        "title": "Image Crop",
        "description": "Crop an image to a rectangle (x, y, width, height).",
        "media": _I,
        "properties": {"x": _INT, "y": _INT, "width": _INT, "height": _INT},
        "required": ["x", "y", "width", "height"],
    },
    {
        "id": "image.format",
        "title": "Image Format Convert",
        "description": "Convert an image between formats (jpeg/png/webp/tiff).",
        "media": _I,
        "properties": {"format": _STR},
        "required": ["format"],
    },
    {
        "id": "image.colour.adjust",
        "title": "Image Colour Adjust",
        "description": "Adjust brightness, contrast and saturation of an image.",
        "media": _I,
        "properties": {"brightness": _INT, "contrast": _INT, "saturation": _INT},
    },
    {
        "id": "audio.trim",
        "title": "Audio Trim",
        "description": "Trim an audio clip to a start/end time range (seconds).",
        "media": _A,
        "properties": {"start": _NUM, "end": _NUM},
        "required": ["start", "end"],
    },
    {
        "id": "audio.concat",
        "title": "Audio Concatenate",
        "description": "Concatenate multiple audio clips into one (input.asset_ids).",
        "media": _A,
        "properties": {"asset_ids": _IDS},
        "required": ["asset_ids"],
    },
    {
        "id": "audio.gain",
        "title": "Audio Gain",
        "description": "Adjust the gain of an audio clip (gain_db).",
        "media": _A,
        "properties": {"gain_db": _NUM},
        "required": ["gain_db"],
    },
    {
        "id": "audio.normalize",
        "title": "Audio Normalize",
        "description": "Normalize loudness to a target integrated level (LUFS).",
        "media": _A,
        "properties": {"target_i": _NUM},
    },
    {
        "id": "audio.fade",
        "title": "Audio Fade",
        "description": "Apply fade-in and/or fade-out (seconds) to an audio clip.",
        "media": _A,
        "properties": {"fade_in": _NUM, "fade_out": _NUM},
    },
    {
        "id": "audio.transcode",
        "title": "Audio Transcode",
        "description": "Transcode audio to a target container/codec.",
        "media": _A,
        "properties": {"format": _STR, "codec": _STR},
        "required": ["format"],
    },
    {
        "id": "audio.denoise",
        "title": "Audio Denoise",
        "description": "Reduce background noise in an audio recording (RNNoise).",
        "media": _A,
        "properties": {"strength": _NUM},
    },
    # --- Composition capabilities ---
    {
        "id": "image.slideshow",
        "title": "Image Slideshow",
        "description": (
            "Build a video slideshow from images (input.asset_ids, in order), "
            "showing each for seconds_per_image."
        ),
        "media": _I,
        "properties": {
            "asset_ids": _IDS,
            "seconds_per_image": _NUM,
            "width": _INT,
            "height": _INT,
        },
        "required": ["asset_ids"],
    },
    {
        "id": "media.titlecard",
        "title": "Title Card",
        "description": (
            "Generate a title-card video clip from text (no input asset needed). "
            "Use as an intro/section card to feed into video.compose."
        ),
        "media": _V,
        "properties": {
            "text": _STR,
            "duration": _NUM,
            "width": _INT,
            "height": _INT,
            "background": _STR,
            "foreground": _STR,
        },
        "required": ["text"],
    },
    {
        "id": "video.subtitle.embed",
        "title": "Embed Subtitles",
        "description": (
            "Embed a subtitle asset (input.subtitle_asset_id) into a video. "
            "mode 'soft' muxes a selectable track; 'burn' renders into the picture."
        ),
        "media": _V,
        "properties": {
            "subtitle_asset_id": _UUID,
            "mode": {"type": "string", "enum": ["soft", "burn"]},
        },
        "required": ["subtitle_asset_id"],
    },
    {
        "id": "audio.mix",
        "title": "Mix Background Audio",
        "description": (
            "Mix a background music/narration asset (input.music_asset_id) under a "
            "video or audio asset. mode: mix | duck | replace."
        ),
        "media": _VA,
        "properties": {
            "music_asset_id": _UUID,
            "music_volume": _NUM,
            "mode": {"type": "string", "enum": ["mix", "duck", "replace"]},
        },
        "required": ["music_asset_id"],
    },
    {
        "id": "video.compose",
        "title": "Compose / Export Timeline",
        "description": (
            "Render a final video from ordered video segments (input.asset_ids) "
            "normalized to width/height, with an optional audio bed "
            "(input.audio_asset_id, audio_mode) and optional burned subtitles "
            "(input.subtitle_asset_id)."
        ),
        "media": _V,
        "properties": {
            "asset_ids": _IDS,
            "width": _INT,
            "height": _INT,
            "audio_asset_id": _UUID,
            "audio_mode": {"type": "string", "enum": ["mix", "duck", "replace"]},
            "music_volume": _NUM,
            "subtitle_asset_id": _UUID,
        },
        "required": ["asset_ids"],
    },
]


CAPABILITIES: list[CapabilityDef] = [
    _def(
        id="audio.transcribe",
        title="Audio Transcription",
        description="Transcribe speech in an audio asset to text.",
        media=[MediaType.audio, MediaType.video],
        cost_class=CostClass.hosted_ai,
        output_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}, "language": {"type": "string"}},
        },
    ),
    _def(
        id="subtitle.autogenerate",
        title="Subtitle Auto-generation",
        description="Generate time-coded subtitles from audio/video.",
        media=[MediaType.audio, MediaType.video],
        cost_class=CostClass.hosted_ai,
        output_schema={"type": "object", "properties": {"srt": {"type": "string"}}},
    ),
    _def(
        id="audio.tts",
        title="Text to Speech",
        description="Synthesize speech audio from text.",
        media=[MediaType.audio],
        cost_class=CostClass.hosted_ai,
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}, "voice": {"type": "string"}},
            "required": ["text"],
        },
        output_schema={"type": "object", "properties": {"audio_url": {"type": "string"}}},
    ),
    _def(
        id="audio.music.generate",
        title="Music Generation",
        description="Generate music audio from a text prompt.",
        media=[MediaType.audio],
        cost_class=CostClass.hosted_ai,
        input_schema={
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
        output_schema={"type": "object", "properties": {"audio_url": {"type": "string"}}},
    ),
    _def(
        id="audio.separate.stems",
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
        title="Background Removal",
        description="Remove the background from an image.",
        media=[MediaType.image],
        cost_class=CostClass.hosted_ai,
        output_schema={"type": "object", "properties": {"image_url": {"type": "string"}}},
    ),
    _def(
        id="image.inpaint",
        title="Image Inpainting",
        description="Fill or replace masked regions of an image.",
        media=[MediaType.image],
        cost_class=CostClass.hosted_ai,
        input_schema={
            "type": "object",
            "properties": {
                "asset_id": {"type": "string", "format": "uuid"},
                "mask_url": {"type": "string"},
                "prompt": {"type": "string"},
            },
        },
        output_schema={"type": "object", "properties": {"image_url": {"type": "string"}}},
    ),
    _def(
        id="image.upscale",
        title="Image Upscale",
        description="Increase image resolution.",
        media=[MediaType.image],
        cost_class=CostClass.hosted_ai,
        output_schema={"type": "object", "properties": {"image_url": {"type": "string"}}},
    ),
    _def(
        id="image.restore.face",
        title="Face Restoration",
        description="Restore and enhance faces in an image.",
        media=[MediaType.image],
        cost_class=CostClass.hosted_ai,
        output_schema={"type": "object", "properties": {"image_url": {"type": "string"}}},
    ),
    _def(
        id="image.caption",
        title="Image Captioning",
        description="Generate a natural-language caption for an image.",
        media=[MediaType.image],
        cost_class=CostClass.hosted_ai,
        output_schema={"type": "object", "properties": {"caption": {"type": "string"}}},
    ),
    _def(
        id="image.tag",
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
        id="video.shot.detect",
        title="Shot Detection",
        description="Detect shot/scene boundaries in a video.",
        media=[MediaType.video],
        cost_class=CostClass.hosted_ai,
        output_schema={
            "type": "object",
            "properties": {"shots": {"type": "array", "items": {"type": "object"}}},
        },
    ),
    _def(
        id="video.upscale",
        title="Video Upscale",
        description="Increase video resolution.",
        media=[MediaType.video],
        cost_class=CostClass.future_gpu,
        output_schema={"type": "object", "properties": {"video_url": {"type": "string"}}},
    ),
    # --- Deterministic local-tool capabilities (FFmpeg / ImageMagick) ---
    # All produce one or more output files, registered as derived Assets and
    # listed under output.outputs[{asset_id, filename, media_type, ...}].
    *(
        _def(
            id=spec["id"],
            title=spec["title"],
            description=spec["description"],
            media=spec["media"],
            cost_class=CostClass.deterministic,
            input_schema={
                "type": "object",
                "properties": spec["properties"],
                "required": spec.get("required", []),
            },
            output_schema=_FILE_OUTPUT_SCHEMA,
        )
        for spec in _DETERMINISTIC_SPECS
    ),
]

CAPABILITIES_BY_ID: dict[str, CapabilityDef] = {c.id: c for c in CAPABILITIES}
