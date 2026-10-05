"""Capability schema (code-defined, not DB-backed)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.models.enums import CostClass, MediaType


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


class CapabilityRead(BaseModel):
    id: str
    title: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    supported_media_types: list[MediaType]
    cost_class: CostClass
    enabled: bool
    applicability: CapabilityApplicability = Field(default_factory=CapabilityApplicability)

    @model_validator(mode="after")
    def primary_media_matches_applicability(self) -> CapabilityRead:
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
