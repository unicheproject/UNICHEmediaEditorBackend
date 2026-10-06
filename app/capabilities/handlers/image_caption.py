"""image.caption handler — routes to the configured inference provider."""

from __future__ import annotations

from app.capabilities.context import HandlerResult, JobContext
from app.capabilities.handlers.base import ProviderBackedHandler
from app.providers.validation import require_caption


class ImageCaptionHandler(ProviderBackedHandler):
    capability_id = "image.caption"

    async def run(self, ctx: JobContext) -> HandlerResult:
        result = await super().run(ctx)
        require_caption(result.data)
        return result
