"""Fallback handler for capabilities that are registered but not yet wired.

Fails truthfully if an old queued Job or programming error reaches the fallback.
"""

from __future__ import annotations

from app.capabilities.context import HandlerResult, JobContext
from app.capabilities.handlers.base import CapabilityHandler
from app.core.errors import CapabilityUnavailableError


class NotImplementedHandler(CapabilityHandler):
    capability_id = "*"

    async def run(self, ctx: JobContext) -> HandlerResult:
        raise CapabilityUnavailableError(
            f"Capability '{ctx.capability_id}' is unavailable (not_implemented)."
        )
