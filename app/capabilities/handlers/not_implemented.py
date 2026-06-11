"""Fallback handler for capabilities that are registered but not yet wired.

Succeeds with a clear not_implemented payload so the registry stays complete
without integrating every provider at once.
"""

from __future__ import annotations

from typing import Any

from app.capabilities.handlers.base import CapabilityHandler
from app.providers.base import InferenceRequest


class NotImplementedHandler(CapabilityHandler):
    capability_id = "*"

    async def run(self, request: InferenceRequest) -> dict[str, Any]:
        return {
            "status": "not_implemented",
            "message": (
                f"Capability '{request.capability_id}' is registered but no handler "
                f"is configured yet."
            ),
        }
