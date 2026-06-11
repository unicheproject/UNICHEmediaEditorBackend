"""Capability registry: lookups over definitions + capability -> handler mapping."""

from __future__ import annotations

from app.capabilities.definitions import CAPABILITIES, CAPABILITIES_BY_ID, CapabilityDef
from app.capabilities.handlers.audio_transcribe import AudioTranscribeHandler
from app.capabilities.handlers.base import CapabilityHandler
from app.capabilities.handlers.image_caption import ImageCaptionHandler
from app.capabilities.handlers.not_implemented import NotImplementedHandler
from app.core.errors import NotFoundError

# Capabilities with a real (provider-backed) handler. Everything else falls
# back to NotImplementedHandler.
_HANDLERS: dict[str, CapabilityHandler] = {
    "image.caption": ImageCaptionHandler(),
    "audio.transcribe": AudioTranscribeHandler(),
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
