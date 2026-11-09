"""Apply a built-in composition template (MED/028) through the normal save path."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.composition import templates
from app.core.errors import NotFoundError
from app.schemas.composition_template import ApplyTemplateRequest, CompositionTemplate
from app.services import compositions


def require_template(template_id: str, version: int | None = None) -> CompositionTemplate:
    """Look a template up in the bundled registry only; never by path."""
    template = (
        templates.latest_template(template_id) if version is None
        else templates.get_template(template_id, version)
    )
    if template is None:
        raise NotFoundError("Composition template not found", code="template_not_found")
    return template


async def apply_template(
    session: AsyncSession,
    project_id: uuid.UUID,
    request: ApplyTemplateRequest,
    *,
    author_subject: str | None,
    if_match: str | None,
    if_none_match: str | None,
) -> tuple[dict, str, bool]:
    """Bind slots, then save as an ordinary revision with the usual preconditions.

    Asset ownership, media types, availability, hashing, ETags, conflicts and the
    transaction are all those of compositions.save_composition. Nothing else is
    created: no Jobs, Assets or projects.
    """
    template = require_template(request.template_id, request.version)
    document = templates.compile_template(template, request.bindings)
    return await compositions.save_composition(
        session, project_id, document,
        author_subject=author_subject, if_match=if_match, if_none_match=if_none_match,
    )
