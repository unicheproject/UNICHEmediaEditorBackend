"""Installation readiness shared by discovery, admission, Agent, and workers.

Applicability remains an input contract. Institutional policy is a later layer.
"""

from __future__ import annotations

import asyncio

from app.core.config import Settings
from app.core.config import settings as default_settings
from app.core.errors import CapabilityUnavailableError
from app.providers.factory import provider_configuration_ready
from app.schemas.capability import (
    CapabilityAvailability,
    CapabilityDefinition,
    CapabilityRead,
    CapabilityReadiness,
    DeploymentReadiness,
    DeploymentRequirement,
    ImplementationReadiness,
    ReadinessReason,
)


def resolve_readiness(
    capability: CapabilityDefinition, *, settings: Settings | None = None
) -> CapabilityReadiness:
    """Pure configuration resolution: no provider calls, CUDA probes, or inference."""
    settings = settings or default_settings
    deployment = DeploymentReadiness.READY
    reason = None
    if capability.implementation == ImplementationReadiness.UNAVAILABLE:
        reason = ReadinessReason.NOT_IMPLEMENTED
    elif capability.deployment_requirement == DeploymentRequirement.INFERENCE_PROVIDER:
        if not provider_configuration_ready(capability.id, settings):
            deployment = DeploymentReadiness.CONFIGURATION_REQUIRED
            reason = ReadinessReason.PROVIDER_NOT_CONFIGURED
    elif (
        capability.deployment_requirement == DeploymentRequirement.CUDA_WORKER
        and not settings.image_upscale_worker_ready
    ):
        deployment = DeploymentReadiness.CONFIGURATION_REQUIRED
        reason = ReadinessReason.CONFIGURATION_REQUIRED
    return CapabilityReadiness(
        implementation=capability.implementation,
        deployment=deployment,
        reason_code=reason,
    )


_UNAVAILABLE_MESSAGES: dict[ReadinessReason, str] = {
    ReadinessReason.NOT_IMPLEMENTED: "{title} is not available yet: it has not been built.",
    ReadinessReason.PROVIDER_NOT_CONFIGURED: (
        "{title} is not available on this server: the AI service it uses has not been "
        "set up. Ask your administrator."
    ),
    ReadinessReason.CONFIGURATION_REQUIRED: (
        "{title} is not available on this server: it needs a setup step that has not "
        "been done. Ask your administrator."
    ),
    ReadinessReason.GPU_UNAVAILABLE: (
        "{title} cannot run right now: the graphics card (GPU) it needs is not "
        "available. Ask your administrator."
    ),
    ReadinessReason.CAPABILITY_DISABLED: "{title} has been switched off on this server.",
}
# image.upscale's configuration step is specifically a GPU worker.
_GPU_CONFIGURATION_MESSAGE = (
    "{title} is not available on this server: it needs a graphics card (GPU) that "
    "has not been set up. Ask your administrator."
)


def unavailable_message(capability: CapabilityDefinition, reason: ReadinessReason) -> str:
    template = _UNAVAILABLE_MESSAGES[reason]
    if (
        reason == ReadinessReason.CONFIGURATION_REQUIRED
        and capability.deployment_requirement == DeploymentRequirement.CUDA_WORKER
    ):
        template = _GPU_CONFIGURATION_MESSAGE
    return template.format(title=capability.title)


def resolve_availability(
    capability: CapabilityDefinition, *, settings: Settings | None = None
) -> CapabilityAvailability:
    readiness = resolve_readiness(capability, settings=settings)
    reason = readiness.reason_code
    if reason is None and not capability.enabled:
        reason = ReadinessReason.CAPABILITY_DISABLED
    executable = (
        capability.enabled
        and readiness.implementation == ImplementationReadiness.AVAILABLE
        and readiness.deployment == DeploymentReadiness.READY
    )
    return CapabilityAvailability(
        executable=executable,
        reason_code=reason,
        message=(
            unavailable_message(capability, reason)
            if not executable and reason is not None else None
        ),
    )


def capability_read(capability: CapabilityDefinition) -> CapabilityRead:
    """Add resolved fields without mutating the canonical code-owned definition."""
    metadata = capability.model_dump(exclude={"implementation", "deployment_requirement"})
    return CapabilityRead(
        **metadata,
        readiness=resolve_readiness(capability),
        availability=resolve_availability(capability),
    )


def require_executable(capability: CapabilityDefinition) -> None:
    availability = resolve_availability(capability)
    if not availability.executable:
        raise CapabilityUnavailableError(
            availability.message or f"{capability.title} is not available.",
            reason_code=availability.reason_code,
        )


async def require_worker_readiness(capability: CapabilityDefinition) -> None:
    """Recheck deployment config and worker-local GPU prerequisites before work."""
    require_executable(capability)
    if capability.deployment_requirement == DeploymentRequirement.CUDA_WORKER:
        from app.tools.realesrgan import check_runtime_readiness

        await asyncio.to_thread(check_runtime_readiness)
