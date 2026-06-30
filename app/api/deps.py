"""Shared FastAPI dependencies for auth + catalogue-backed project access.

`get_current_principal` (validate the token) is applied globally to the whole
``/api/v1`` router. The guards here additionally enforce *project-level*
authorization via the catalogue, lazy-provisioning the companion row:

- `require_project` — for routes with ``project_id`` in the path.
- `require_asset_access` / `require_job_access` — resolve the owning project of
  a resource addressed by its own id, then authorize it.
"""

from __future__ import annotations

import uuid

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.core.security import Principal, get_current_principal
from app.models.asset import Asset
from app.models.job import Job
from app.models.project import Project
from app.schemas.job import JobCreate
from app.services import assets as assets_svc
from app.services import jobs as jobs_svc
from app.services import projects as projects_svc
from app.services.catalogue_client import CatalogueClient, get_catalogue_client


async def require_project(
    project_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
) -> Project:
    return await projects_svc.require_project_access(
        session, catalogue, principal, project_id
    )


async def require_asset_access(
    asset_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
) -> Asset:
    asset = await assets_svc.get_asset(session, asset_id)
    await projects_svc.require_project_access(
        session, catalogue, principal, asset.project_id
    )
    return asset


async def require_job_access(
    job_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
) -> Job:
    job = await jobs_svc.get_job(session, job_id)
    if job.project_id is not None:
        await projects_svc.require_project_access(
            session, catalogue, principal, job.project_id
        )
    return job


async def require_job_create_access(
    data: JobCreate,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
) -> JobCreate:
    """Authorize the project a new job will run against (from body or its asset)."""
    project_id = data.project_id
    if project_id is None and data.asset_id is not None:
        asset = await assets_svc.get_asset(session, data.asset_id)
        project_id = asset.project_id
    if project_id is not None:
        await projects_svc.require_project_access(
            session, catalogue, principal, project_id
        )
    return data
