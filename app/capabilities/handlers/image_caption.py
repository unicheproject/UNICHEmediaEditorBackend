"""image.caption handler — routes to the configured inference provider."""

from __future__ import annotations

from app.capabilities.handlers.base import ProviderBackedHandler


class ImageCaptionHandler(ProviderBackedHandler):
    capability_id = "image.caption"
