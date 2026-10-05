"""Provider selection. The only place that knows which provider is active."""

from __future__ import annotations

from app.core.config import Settings
from app.core.config import settings as default_settings
from app.core.errors import ProviderError
from app.providers.base import BaseInferenceProvider
from app.providers.http import HTTPInferenceProvider
from app.providers.mock import MockInferenceProvider
from app.providers.openrouter import OpenRouterInferenceProvider


def get_provider(settings: Settings | None = None) -> BaseInferenceProvider:
    settings = settings or default_settings
    provider = settings.inference_provider.lower()
    if provider == "mock":
        return MockInferenceProvider()
    if provider == "http":
        return HTTPInferenceProvider(settings)
    if provider == "openrouter":
        return OpenRouterInferenceProvider(settings)
    raise ProviderError(f"Unknown INFERENCE_PROVIDER '{settings.inference_provider}'")


def provider_configuration_ready(capability_id: str, settings: Settings) -> bool:
    """Resolve deployment configuration without network I/O or secret disclosure."""
    try:
        return get_provider(settings).configuration_ready(capability_id)
    except ProviderError:
        return False
