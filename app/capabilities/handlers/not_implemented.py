"""Fallback handler for capabilities that are registered but not yet wired.

Succeeds with a clear not_implemented payload so the registry stays complete
without integrating every provider at once.
"""

from __future__ import annotations

from app.capabilities.context import HandlerResult, JobContext
from app.capabilities.handlers.base import CapabilityHandler


class NotImplementedHandler(CapabilityHandler):
    capability_id = "*"

    async def run(self, ctx: JobContext) -> HandlerResult:
        return HandlerResult(
            data={
                "status": "not_implemented",
                "message": (
                    f"Capability '{ctx.capability_id}' is registered but no handler "
                    f"is configured yet."
                ),
            },
            outputs=[],
        )
