"""Deterministic offline inference provider (default).

Produces stable, fake output derived from asset metadata so the full job
pipeline works with no API keys and tests are reproducible.
"""

from __future__ import annotations

from typing import Any

from app.providers.base import BaseInferenceProvider, InferenceRequest


class MockInferenceProvider(BaseInferenceProvider):
    name = "mock"

    async def infer(self, request: InferenceRequest) -> dict[str, Any]:
        meta = request.asset_meta
        original = meta.get("original_filename", "asset")
        media_type = meta.get("media_type", "unknown")

        if request.capability_id == "image.caption":
            caption = (
                f"A {media_type} titled '{original}' showing a cultural-heritage "
                f"scene (mock caption)."
            )
            return {"caption": caption, "provider": self.name}

        if request.capability_id == "audio.transcribe":
            transcript = (
                f"[mock transcript for '{original}'] This is a deterministic "
                f"placeholder transcription generated offline."
            )
            return {
                "text": transcript,
                "language": "en",
                "provider": self.name,
            }

        # Generic deterministic fallback for any other capability.
        return {
            "result": f"mock output for {request.capability_id} on '{original}'",
            "provider": self.name,
        }
