"""Generic capability input-scope admission rules."""

from __future__ import annotations

import uuid

from app.core.errors import ValidationError
from app.schemas.capability import (
    CapabilityMetadata,
    PrimaryAssetApplicability,
    ProjectContextApplicability,
)


def validate_job_applicability(
    capability: CapabilityMetadata,
    *,
    asset_id: uuid.UUID | None,
    requested_project_id: uuid.UUID | None,
    resolved_project_id: uuid.UUID | None = None,
    project_resolution_complete: bool = False,
) -> None:
    """Enforce primary-Asset and project-context semantics without ID checks."""
    applicability = capability.applicability
    validate_primary_asset_applicability(capability, has_primary_asset=asset_id is not None)
    if (
        applicability.project_context == ProjectContextApplicability.REQUIRED
        and requested_project_id is None
    ):
        raise ValidationError(
            f"Capability '{capability.id}' requires an explicit project_id"
        )
    if (
        project_resolution_complete
        and resolved_project_id is None
    ):
        raise ValidationError(
            f"Capability '{capability.id}' requires project context from project_id "
            "or its Asset references"
        )


def validate_primary_asset_applicability(
    capability: CapabilityMetadata, *, has_primary_asset: bool
) -> None:
    """Apply the registry's primary-Asset rule for Jobs and Agent plan steps."""
    primary_asset = capability.applicability.primary_asset
    if primary_asset == PrimaryAssetApplicability.REQUIRED and not has_primary_asset:
        raise ValidationError(
            f"Capability '{capability.id}' requires a primary asset_id"
        )
    if primary_asset == PrimaryAssetApplicability.FORBIDDEN and has_primary_asset:
        raise ValidationError(
            f"Capability '{capability.id}' does not accept a primary asset_id"
        )
