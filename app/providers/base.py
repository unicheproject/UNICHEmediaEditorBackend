"""Inference provider abstraction.

Providers turn a (capability, payload, optional file) into an output dict.
Provider selection and configuration checks stay behind the provider boundary.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceRequest:
    capability_id: str
    # Free-form job input (e.g. prompt, options) plus capability context.
    payload: dict[str, Any] = field(default_factory=dict)
    # Local filesystem path to the input asset, if one is attached.
    file_path: str | None = None
    # Metadata about the asset, useful for deterministic mock output.
    asset_meta: dict[str, Any] = field(default_factory=dict)


class BaseInferenceProvider(abc.ABC):
    """Common interface for all inference providers."""

    name: str = "base"

    def configuration_ready(self, capability_id: str) -> bool:
        """Cheap local configuration check; never inference or a live health check.

        Providers opt in to real execution. The mock intentionally stays false.
        """
        return False

    @abc.abstractmethod
    async def infer(self, request: InferenceRequest) -> dict[str, Any]:
        """Run inference for a capability and return an output dict.

        Implementations should raise app.core.errors.ProviderError on failure.
        """
        raise NotImplementedError
