"""Read-only listing of the built-in composition templates (MED/028)."""

from __future__ import annotations

from fastapi import APIRouter, Path

from app.composition import templates
from app.schemas.composition_template import (
    TEMPLATE_ID_PATTERN,
    CompositionTemplate,
    CompositionTemplateSummary,
)
from app.services.composition_templates import require_template

# Validate every bundled template at startup; an invalid one fails loudly.
templates.load_registry()

router = APIRouter(prefix="/composition-templates", tags=["composition"])

_TemplateId = Path(pattern=TEMPLATE_ID_PATTERN, max_length=64)


@router.get("", response_model=list[CompositionTemplateSummary])
async def list_templates() -> list[CompositionTemplateSummary]:
    return [
        CompositionTemplateSummary(
            template_id=t.template_id, version=t.version, name=t.name,
            description=t.description, slots=dict(t.slots),
            suggested_export_preset=t.suggested_export_preset,
        )
        for t in templates.latest_templates()
    ]


@router.get("/{template_id}", response_model=CompositionTemplate)
async def get_latest_template(template_id: str = _TemplateId) -> CompositionTemplate:
    return require_template(template_id)


@router.get("/{template_id}/versions/{version}", response_model=CompositionTemplate)
async def get_template_version(
    template_id: str = _TemplateId, version: int = Path(ge=1)
) -> CompositionTemplate:
    return require_template(template_id, version)
