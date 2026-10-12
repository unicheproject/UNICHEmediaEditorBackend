"""Capability registry endpoints (read-only)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.capabilities import registry
from app.capabilities.readiness import capability_read
from app.schemas.capability import CapabilityRead

router = APIRouter(prefix="/capabilities", tags=["capabilities"])


@router.get("", response_model=list[CapabilityRead])
async def list_capabilities(
    include_disabled: bool = Query(
        default=False,
        description=(
            "Include code-defined capabilities disabled by application policy. "
            "Discovery does not make a capability executable."
        ),
    ),
) -> list[CapabilityRead]:
    capabilities = registry.list_defined() if include_disabled else registry.list_enabled()
    return [capability_read(cap) for cap in capabilities]


@router.get("/{capability_id}", response_model=CapabilityRead)
async def get_capability(capability_id: str) -> CapabilityRead:
    return capability_read(registry.get(capability_id))
