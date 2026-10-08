"""Capability schema (code-defined, not DB-backed)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.models.enums import AiContentEffect, CostClass, MediaType


class PrimaryAssetApplicability(StrEnum):
    """Whether a Job's top-level primary Asset is accepted by a capability."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    FORBIDDEN = "forbidden"


class ProjectContextApplicability(StrEnum):
    """Whether project scope must be named or can be inferred from inputs."""

    REQUIRED = "required"
    DERIVED = "derived"


class CapabilityApplicability(BaseModel):
    """Structural input-scope contract, independent of readiness/policy."""

    primary_asset: PrimaryAssetApplicability = PrimaryAssetApplicability.REQUIRED
    project_context: ProjectContextApplicability = ProjectContextApplicability.DERIVED


class ImplementationReadiness(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class DeploymentReadiness(StrEnum):
    READY = "ready"
    CONFIGURATION_REQUIRED = "configuration_required"
    UNAVAILABLE = "unavailable"


class ReadinessReason(StrEnum):
    NOT_IMPLEMENTED = "not_implemented"
    PROVIDER_NOT_CONFIGURED = "provider_not_configured"
    GPU_UNAVAILABLE = "gpu_unavailable"
    CONFIGURATION_REQUIRED = "configuration_required"
    CAPABILITY_DISABLED = "capability_disabled"


class DeploymentRequirement(StrEnum):
    """Code-owned execution requirements; independent of input applicability."""

    LOCAL = "local"
    INFERENCE_PROVIDER = "inference_provider"
    CUDA_WORKER = "cuda_worker"


class CapabilityReadiness(BaseModel):
    implementation: ImplementationReadiness
    deployment: DeploymentReadiness
    reason_code: ReadinessReason | None = None


class CapabilityAvailability(BaseModel):
    executable: bool
    reason_code: ReadinessReason | None = None


class CapabilityMetadata(BaseModel):
    id: str
    title: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    supported_media_types: list[MediaType]
    cost_class: CostClass
    enabled: bool
    ai_content_effect: AiContentEffect = Field(
        default=AiContentEffect.none,
        description=(
            "Direct AI generation/modification by this operation, independent of cost "
            "or Agent initiation."
        ),
    )
    client_materialization_media_types: list[MediaType] = Field(
        default_factory=list,
        description=(
            "Upload media types eligible as possibly edited derivatives of a successful Job result."
        ),
    )
    applicability: CapabilityApplicability = Field(default_factory=CapabilityApplicability)

    @model_validator(mode="after")
    def primary_media_matches_applicability(self) -> CapabilityMetadata:
        if self.applicability.primary_asset == PrimaryAssetApplicability.FORBIDDEN:
            if self.supported_media_types:
                raise ValueError(
                    "A capability that forbids a primary Asset must not advertise "
                    "supported_media_types"
                )
        elif not self.supported_media_types:
            raise ValueError(
                "A capability that accepts a primary Asset must advertise "
                "supported_media_types"
            )
        return self


class CapabilityDefinition(CapabilityMetadata):
    """Static code-owned declaration. Deployment readiness is resolved separately."""

    implementation: ImplementationReadiness
    deployment_requirement: DeploymentRequirement = DeploymentRequirement.LOCAL


class CapabilityRead(CapabilityMetadata):
    """Public metadata with the installation's current readiness resolution."""

    readiness: CapabilityReadiness
    availability: CapabilityAvailability
