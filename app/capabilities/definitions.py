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
        id="audio.denoise",
        title="Audio Denoise",
        description="Reduce background noise in an audio asset.",
        media=[MediaType.audio, MediaType.video],
        cost_class=CostClass.hosted_ai,
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
]

CAPABILITIES_BY_ID: dict[str, CapabilityDef] = {c.id: c for c in CAPABILITIES}
