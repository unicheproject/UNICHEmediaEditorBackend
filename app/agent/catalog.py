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
    """Compact execution/semantic contract for embedding in the planner prompt."""
    lines = []
    for c in build_catalog():
        schema = c["input_schema"]
        required = set(schema.get("required", []))
        params = [
            _parameter_text(name, prop, required=name in required)
            for name, prop in sorted((schema.get("properties") or {}).items())
        ]
        media = "/".join(c["supported_media_types"]) or "none"
        lines.append(
            f"- {c['capability_id']} ({media}): {c['description']} params: [{'; '.join(params)}]"
        )
    return "\n".join(lines)


def _parameter_text(name: str, schema: dict[str, Any], *, required: bool) -> str:
    """Serialize one property without exposing UI-only or irrelevant schema data."""
    type_name = str(schema.get("type", "any"))
    if type_name == "array":
        item_schema = schema.get("items", {})
        item_type = str(item_schema.get("type", "any"))
        if item_schema.get("format"):
            item_type = f"{item_type}/{item_schema['format']}"
        type_name = f"array<{item_type}>"
    if schema.get("format"):
        type_name = f"{type_name}/{schema['format']}"

    facts = [type_name, "required" if required else "optional"]
    for key, label in (
        ("default", "default"),
        ("minimum", "min"),
        ("exclusiveMinimum", "exclusive_min"),
        ("maximum", "max"),
        ("minItems", "min_items"),
        ("maxItems", "max_items"),
    ):
        if key in schema:
            facts.append(f"{label}={schema[key]!r}")
    if "enum" in schema:
        enum_semantics = schema.get("x-enum-descriptions", {})
        options = [
            f"{value} ({enum_semantics[value]})" if value in enum_semantics else str(value)
            for value in schema["enum"]
        ]
        facts.append(f"enum={options!r}")
    option_constraints = schema.get("x-option-constraints")
    if option_constraints:
        dependency = option_constraints["depends_on"]
        allowed_values = option_constraints["allowed_values"]
        facts.append(
            f"choices_when_{dependency}={allowed_values!r}"
        )
    if schema.get("x-unit"):
        facts.append(f"unit={schema['x-unit']}")
    media = schema.get("x-media-asset")
    if media:
        facts.append(f"asset_role={media['role']}")
        facts.append(f"asset_media={media.get('media_types', [])!r}")

    description = schema.get("description")
    suffix = f" — {description}" if description else ""
    return f"{name}: {' '.join(facts)}{suffix}"
