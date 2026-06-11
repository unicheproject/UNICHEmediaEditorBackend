"""Project service: CRUD with soft deletes."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectUpdate


async def create_project(session: AsyncSession, data: ProjectCreate) -> Project:
    project = Project(name=data.name, description=data.description)
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return project


async def list_projects(session: AsyncSession) -> list[Project]:
    result = await session.execute(
        select(Project)
        .where(Project.deleted_at.is_(None))
        .order_by(Project.created_at.desc())
    )
    return list(result.scalars().all())


async def get_project(session: AsyncSession, project_id: uuid.UUID) -> Project:
    project = await session.get(Project, project_id)
    if project is None or project.deleted_at is not None:
        raise NotFoundError(f"Project '{project_id}' not found")
    return project


async def update_project(
    session: AsyncSession, project_id: uuid.UUID, data: ProjectUpdate
) -> Project:
    project = await get_project(session, project_id)
    fields = data.model_dump(exclude_unset=True)
    for key, value in fields.items():
        setattr(project, key, value)
    await session.commit()
    await session.refresh(project)
    return project


async def soft_delete_project(session: AsyncSession, project_id: uuid.UUID) -> None:
    project = await get_project(session, project_id)
    project.deleted_at = datetime.now(UTC)
    await session.commit()
