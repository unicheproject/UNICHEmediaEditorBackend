"""audio.transcribe handler — routes to the configured inference provider."""

from __future__ import annotations

from app.capabilities.context import HandlerResult, JobContext
from app.capabilities.handlers.base import ProviderBackedHandler
from app.providers.validation import require_transcript


class AudioTranscribeHandler(ProviderBackedHandler):
    capability_id = "audio.transcribe"

    async def run(self, ctx: JobContext) -> HandlerResult:
        result = await super().run(ctx)
        require_transcript(result.data)
        return result
