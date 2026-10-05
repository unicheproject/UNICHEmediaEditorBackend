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


def resolve_availability(
    capability: CapabilityDefinition, *, settings: Settings | None = None
) -> CapabilityAvailability:
    readiness = resolve_readiness(capability, settings=settings)
    reason = readiness.reason_code
    if reason is None and not capability.enabled:
        reason = ReadinessReason.CAPABILITY_DISABLED
    return CapabilityAvailability(
        executable=(
            capability.enabled
            and readiness.implementation == ImplementationReadiness.AVAILABLE
            and readiness.deployment == DeploymentReadiness.READY
        ),
        reason_code=reason,
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
            f"Capability '{capability.id}' is unavailable "
            f"({availability.reason_code})."
        )


async def require_worker_readiness(capability: CapabilityDefinition) -> None:
    """Recheck deployment config and worker-local GPU prerequisites before work."""
    require_executable(capability)
    if capability.deployment_requirement == DeploymentRequirement.CUDA_WORKER:
        from app.tools.realesrgan import check_runtime_readiness

        await asyncio.to_thread(check_runtime_readiness)
