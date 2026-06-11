"""Capability handler interface.

A handler encapsulates how a capability turns a Job into output. Handlers
delegate hosted-inference work to the provider abstraction, so swapping
providers never touches handler call sites in the API.
"""

from __future__ import annotations

import abc
from typing import Any

from app.providers.base import InferenceRequest
from app.providers.factory import get_provider


class CapabilityHandler(abc.ABC):
    capability_id: str

    @abc.abstractmethod
    async def run(self, request: InferenceRequest) -> dict[str, Any]:
        raise NotImplementedError


class ProviderBackedHandler(CapabilityHandler):
    """Handler that simply routes to the configured inference provider."""

    async def run(self, request: InferenceRequest) -> dict[str, Any]:
        provider = get_provider()
        return await provider.infer(request)
