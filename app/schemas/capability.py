"""Capability schema (code-defined, not DB-backed)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.models.enums import CostClass, MediaType


class CapabilityRead(BaseModel):
    id: str
    title: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    supported_media_types: list[MediaType]
    cost_class: CostClass
    enabled: bool
