"""Project endpoints.

Projects are owned by the Catalogue. These routes proxy create/edit/delete to
it with the user's token and keep a local companion row; the picker is built
live from the catalogue. See ``app.services.projects``.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_project
from app.core.database import get_session
from app.core.security import Principal, get_current_principal
from app.models.project import Project
from app.schemas.project import (
    ProjectCreate,
    ProjectListItem,
    ProjectRead,
    ProjectUpdate,
)
from app.services import projects as svc
from app.services.catalogue_client import CatalogueClient, get_catalogue_client

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
async def create_project(
    data: ProjectCreate,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
) -> ProjectRead:
    project = await svc.create_project(session, catalogue, principal, data)
    return ProjectRead.model_validate(project)


@router.get("", response_model=list[ProjectListItem])
async def list_projects(
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
) -> list[ProjectListItem]:
    return await svc.list_projects_for_user(session, catalogue, principal)


@router.get("/{project_id}", response_model=ProjectRead)
async def get_project(project: Project = Depends(require_project)) -> ProjectRead:
    return ProjectRead.model_validate(project)


@router.patch("/{project_id}", response_model=ProjectRead)
async def update_project(
    project_id: uuid.UUID,
    data: ProjectUpdate,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
) -> ProjectRead:
    project = await svc.update_project(session, catalogue, principal, project_id, data)
    return ProjectRead.model_validate(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
) -> None:
    await svc.delete_project(session, catalogue, principal, project_id)
