"""audio.transcribe handler — routes to the configured inference provider."""

from __future__ import annotations

from app.capabilities.handlers.base import ProviderBackedHandler


class AudioTranscribeHandler(ProviderBackedHandler):
    capability_id = "audio.transcribe"
