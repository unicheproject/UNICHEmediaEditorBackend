"""Project-scoped Composition read and conditional-save endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Header, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_project
from app.core.database import get_session
from app.core.security import Principal, get_current_principal
from app.models.project import Project
from app.schemas.composition import CompositionDocument, CompositionRevisionRead
from app.services import compositions as svc

router = APIRouter(
    prefix="/projects/{project_id}/composition",
    tags=["composition"],
)


@router.get("", response_model=CompositionRevisionRead)
async def get_current_composition(
    response: Response,
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _project: Project = Depends(require_project),
) -> dict:
    composition, etag = await svc.get_current_composition(session, project_id)
    response.headers["ETag"] = etag
    return composition


@router.put(
    "",
    response_model=CompositionRevisionRead,
    status_code=status.HTTP_200_OK,
)
async def save_composition(
    data: CompositionDocument,
    response: Response,
    project_id: uuid.UUID,
    if_match: str | None = Header(default=None, alias="If-Match"),
    if_none_match: str | None = Header(default=None, alias="If-None-Match"),
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    _project: Project = Depends(require_project),
) -> dict:
    saved, etag, created = await svc.save_composition(
        session,
        project_id,
        data,
        author_subject=principal.subject,
        if_match=if_match,
        if_none_match=if_none_match,
    )
    response.headers["ETag"] = etag
    if created and saved["revision_number"] == 1:
        response.status_code = status.HTTP_201_CREATED
    return saved


@router.get(
    "/revisions/{revision_id}",
    response_model=CompositionRevisionRead,
)
async def get_composition_revision(
    project_id: uuid.UUID,
    revision_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _project: Project = Depends(require_project),
) -> dict:
    return await svc.get_composition_revision(session, project_id, revision_id)
