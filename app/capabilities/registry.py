"""Capability registry: lookups over definitions + capability -> handler mapping."""

from __future__ import annotations

from app.capabilities.definitions import CAPABILITIES, CAPABILITIES_BY_ID, CapabilityDef
from app.capabilities.handlers import audio_ops, compose_ops, image_ops, video_ops
from app.capabilities.handlers.audio_transcribe import AudioTranscribeHandler
from app.capabilities.handlers.audio_tts import AudioTtsHandler
from app.capabilities.handlers.base import CapabilityHandler, ProviderBackedHandler
from app.capabilities.handlers.image_caption import ImageCaptionHandler
from app.capabilities.handlers.not_implemented import NotImplementedHandler
from app.capabilities.handlers.project_export import ProjectExportHandler
from app.core.errors import NotFoundError
from app.schemas.capability import DeploymentRequirement, ImplementationReadiness

# Deterministic local-tool handlers (FFmpeg / ImageMagick / Real-ESRGAN / PySceneDetect).
_TOOL_HANDLERS: list[CapabilityHandler] = [
    video_ops.VideoTrimHandler(),
    video_ops.VideoSplitHandler(),
    video_ops.VideoConcatHandler(),
    video_ops.VideoTranscodeHandler(),
    video_ops.VideoMuteHandler(),
    video_ops.VideoCropHandler(),
    video_ops.VideoResizeHandler(),
    video_ops.VideoThumbnailHandler(),
    video_ops.VideoShotDetectHandler(),
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
    audio_ops.VideoAudioExtractHandler(),
    compose_ops.ImageSlideshowHandler(),
    compose_ops.ImageKenburnsHandler(),
    compose_ops.MediaTitlecardHandler(),
    compose_ops.VideoLowerthirdHandler(),
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
    "project.export": ProjectExportHandler(),
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


def validate_implementation_registry() -> None:
    """Fail startup on declaration/handler drift, before advertising definitions."""
    if len(CAPABILITIES_BY_ID) != len(CAPABILITIES):
        raise RuntimeError("Duplicate capability definitions")
    for capability in CAPABILITIES:
        handler = _HANDLERS.get(capability.id)
        implemented = handler is not None and not isinstance(handler, NotImplementedHandler)
        declared = capability.implementation == ImplementationReadiness.AVAILABLE
        if implemented != declared:
            raise RuntimeError(f"Implementation readiness/handler mismatch: {capability.id}")
        if handler is not None and handler.capability_id != capability.id:
            raise RuntimeError(f"Handler identity mismatch: {capability.id}")
        if (
            isinstance(handler, ProviderBackedHandler)
            and capability.deployment_requirement != DeploymentRequirement.INFERENCE_PROVIDER
        ):
            raise RuntimeError(f"Provider deployment requirement missing: {capability.id}")
    if set(_HANDLERS) - set(CAPABILITIES_BY_ID):
        raise RuntimeError("Handlers without capability definitions")


validate_implementation_registry()
