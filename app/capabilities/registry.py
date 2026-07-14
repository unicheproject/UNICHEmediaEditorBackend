"""Capability registry: lookups over definitions + capability -> handler mapping."""

from __future__ import annotations

from app.capabilities.definitions import CAPABILITIES, CAPABILITIES_BY_ID, CapabilityDef
from app.capabilities.handlers import audio_ops, compose_ops, image_ops, video_ops
from app.capabilities.handlers.audio_transcribe import AudioTranscribeHandler
from app.capabilities.handlers.audio_tts import AudioTtsHandler
from app.capabilities.handlers.base import CapabilityHandler
from app.capabilities.handlers.image_caption import ImageCaptionHandler
from app.capabilities.handlers.not_implemented import NotImplementedHandler
from app.core.errors import NotFoundError

# Deterministic local-tool handlers (FFmpeg / ImageMagick / Real-ESRGAN).
_TOOL_HANDLERS: list[CapabilityHandler] = [
    video_ops.VideoTrimHandler(),
    video_ops.VideoSplitHandler(),
    video_ops.VideoConcatHandler(),
    video_ops.VideoTranscodeHandler(),
    video_ops.VideoMuteHandler(),
    video_ops.VideoCropHandler(),
    video_ops.VideoResizeHandler(),
    video_ops.VideoThumbnailHandler(),
    image_ops.ImageResizeHandler(),
    image_ops.ImageCropHandler(),
    image_ops.ImageFormatHandler(),
    image_ops.ImageColourAdjustHandler(),
    image_ops.ImageUpscaleHandler(),
    audio_ops.AudioTrimHandler(),
    audio_ops.AudioConcatHandler(),
    audio_ops.AudioGainHandler(),
    audio_ops.AudioNormalizeHandler(),
    audio_ops.AudioFadeHandler(),
    audio_ops.AudioDenoiseHandler(),
    audio_ops.AudioTranscodeHandler(),
    compose_ops.ImageSlideshowHandler(),
    compose_ops.MediaTitlecardHandler(),
    compose_ops.VideoSubtitleEmbedHandler(),
    compose_ops.AudioMixHandler(),
    compose_ops.VideoComposeHandler(),
]

# Capabilities with a real handler (provider-backed or local-tool). Everything
# else falls back to NotImplementedHandler.
_HANDLERS: dict[str, CapabilityHandler] = {
    "image.caption": ImageCaptionHandler(),
    "audio.transcribe": AudioTranscribeHandler(),
    "audio.tts": AudioTtsHandler(),
    **{h.capability_id: h for h in _TOOL_HANDLERS},
}
_FALLBACK_HANDLER = NotImplementedHandler()


def list_enabled() -> list[CapabilityDef]:
    return [c for c in CAPABILITIES if c.enabled]


def exists(capability_id: str) -> bool:
    return capability_id in CAPABILITIES_BY_ID


def get(capability_id: str) -> CapabilityDef:
    cap = CAPABILITIES_BY_ID.get(capability_id)
    if cap is None:
        raise NotFoundError(f"Capability '{capability_id}' not found")
    return cap


def get_handler(capability_id: str) -> CapabilityHandler:
    return _HANDLERS.get(capability_id, _FALLBACK_HANDLER)
