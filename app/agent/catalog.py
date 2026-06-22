"""Build the LLM tool catalog from the capability registry.

The registry is the single source of truth: each enabled capability's id,
description, input_schema and supported media types become a tool spec the
planner can choose from. Adding a capability automatically extends the agent.
"""

from __future__ import annotations

from typing import Any

from app.capabilities import registry
from app.core.config import settings
from app.models.enums import CostClass


def build_catalog() -> list[dict[str, Any]]:
    """A compact, model-friendly description of every enabled capability.

    When ``agent_deterministic_only`` is set, hosted-AI / GPU capabilities are
    excluded so the planner can only choose deterministic local-tool ops.
    """
    caps = registry.list_enabled()
    if settings.agent_deterministic_only:
        caps = [c for c in caps if c.cost_class == CostClass.deterministic]
    catalog = []
    for cap in caps:
        catalog.append(
            {
                "capability_id": cap.id,
                "title": cap.title,
                "description": cap.description,
                "supported_media_types": [m.value for m in cap.supported_media_types],
                "cost_class": cap.cost_class.value,
                "input_schema": cap.input_schema,
            }
        )
    return catalog


def catalog_text() -> str:
    """Human/LLM-readable catalog lines for embedding in a system prompt."""
    lines = []
    for c in build_catalog():
        props = ", ".join(sorted((c["input_schema"].get("properties") or {}).keys()))
        media = "/".join(c["supported_media_types"]) or "none"
        lines.append(
            f"- {c['capability_id']} ({media}): {c['description']} params: [{props}]"
        )
    return "\n".join(lines)
