"""Capability registry endpoints (read-only)."""

from __future__ import annotations

from fastapi import APIRouter

from app.capabilities import registry
from app.schemas.capability import CapabilityRead

router = APIRouter(prefix="/capabilities", tags=["capabilities"])


@router.get("", response_model=list[CapabilityRead])
async def list_capabilities() -> list[CapabilityRead]:
    return registry.list_enabled()


@router.get("/{capability_id}", response_model=CapabilityRead)
async def get_capability(capability_id: str) -> CapabilityRead:
    return registry.get(capability_id)
