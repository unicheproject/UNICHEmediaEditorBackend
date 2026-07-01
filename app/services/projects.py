"""Project service.

The Catalogue is the source of truth and authorization authority. Local rows
are *companion* rows keyed by the catalogue's project UUID:

- **access + lazy-JIT** (`require_project_access`): a single `GET /projects/{id}`
  with the user's token is both the access check and the provisioning source;
  a 404 means no-access-or-deleted and soft-deletes the local row.
- **create-up** (`create_project`): create in the catalogue, then store the
  companion row keyed by the returned UUID (plus editor-local description).
- **picker** (`list_projects_for_user`): built live from the user's token.
- **edit/delete**: proxied to the catalogue (manager-only there); description
  edits stay local.

`get_project` remains a plain local-row fetch used internally by the asset/job
services (the companion row already exists by then, via lazy-JIT).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import NotFoundError
from app.core.security import Principal
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectListItem, ProjectUpdate
from app.services.catalogue_client import CatalogueClient


def _uuid_or_none(value: str | None) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except (ValueError, TypeError):
        return None


# --- Local-row helpers -----------------------------------------------------


async def get_project(session: AsyncSession, project_id: uuid.UUID) -> Project:
    """Fetch the local companion row; raise if missing/soft-deleted."""
    project = await session.get(Project, project_id)
    if project is None or project.deleted_at is not None:
        raise NotFoundError(f"Project '{project_id}' not found")
    return project


async def upsert_companion(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    org_id: uuid.UUID | None,
    name: str,
) -> Project:
    """Create or refresh the companion row for a catalogue project (idempotent)."""
    project = await session.get(Project, project_id)
    if project is None:
        project = Project(id=project_id, org_id=org_id, name=name)
        session.add(project)
    else:
        project.name = name
        project.org_id = org_id
        if project.deleted_at is not None:
            # Catalogue says it exists and the user can access it → undelete.
            project.deleted_at = None
    await session.commit()
    await session.refresh(project)
    return project


async def soft_delete_local(session: AsyncSession, project_id: uuid.UUID) -> None:
    project = await session.get(Project, project_id)
    if project is not None and project.deleted_at is None:
        project.deleted_at = datetime.now(UTC)
        await session.commit()


async def _local_descriptions(
    session: AsyncSession, ids: list[uuid.UUID]
) -> dict[uuid.UUID, str | None]:
    if not ids:
        return {}
    result = await session.execute(
        select(Project.id, Project.description).where(
            Project.id.in_(ids), Project.deleted_at.is_(None)
        )
    )
    return {row.id: row.description for row in result.all()}


# --- Catalogue-backed operations -------------------------------------------


async def require_project_access(
    session: AsyncSession,
    catalogue: CatalogueClient,
    principal: Principal,
    project_id: uuid.UUID,
) -> Project:
    """Authorize via the catalogue (user token) and lazy-provision the row.

    Returns the companion row on success; raises NotFoundError when the
    catalogue denies access or the project is gone (also soft-deleting any
    stale local row).
    """
    cat = await catalogue.get_project(str(project_id), principal.token)
    if cat is None:
        await soft_delete_local(session, project_id)
        raise NotFoundError(f"Project '{project_id}' not found")
    return await upsert_companion(
        session,
        project_id=project_id,
        org_id=_uuid_or_none(cat.get("orgId")),
        name=cat.get("name") or "",
    )


async def create_project(
    session: AsyncSession,
    catalogue: CatalogueClient,
    principal: Principal,
    data: ProjectCreate,
) -> Project:
    """Create-up: create in the catalogue, then store the companion row."""
    cat = await catalogue.create_project(
        str(data.org_id), data.name, data.slug, principal.token
    )
    project = await upsert_companion(
        session,
        project_id=uuid.UUID(cat["id"]),
        org_id=_uuid_or_none(cat.get("orgId")) or data.org_id,
        name=cat.get("name") or data.name,
    )
    if data.description is not None:
        project.description = data.description
        await session.commit()
        await session.refresh(project)
    return project


async def list_projects_for_user(
    session: AsyncSession,
    catalogue: CatalogueClient,
    principal: Principal,
) -> list[ProjectListItem]:
    """Live picker: media-editor projects the user can access (user token).

    The org set comes from ``GET /organisations`` (the catalogue's ``listVisible``),
    which already encodes the authorization rule for every role: a platform admin
    sees all orgs, a manager sees managed orgs, a curator sees the orgs of their
    project memberships. Each org's project list is then itself scoped by the
    catalogue to what the caller may see. We filter to this tool's slug.
    """
    orgs = await catalogue.list_organisations(principal.token)

    found: dict[str, dict] = {}
    for org in orgs:
        org_id = org.get("id")
        if not org_id:
            continue
        for project in await catalogue.list_org_projects(org_id, principal.token):
            tool = project.get("tool") or {}
            if tool.get("slug") == settings.tool_slug:
                found[project["id"]] = project

    descriptions = await _local_descriptions(
        session, [uuid.UUID(i) for i in found]
    )
    items = [
        ProjectListItem(
            id=uuid.UUID(p["id"]),
            org_id=_uuid_or_none(p.get("orgId")),
            name=p.get("name") or "",
            slug=p.get("slug"),
            status=p.get("status"),
            description=descriptions.get(uuid.UUID(p["id"])),
            created_at=p.get("createdAt"),
            updated_at=p.get("updatedAt"),
        )
        for p in found.values()
    ]
    items.sort(key=lambda item: item.name.lower())
    return items


async def update_project(
    session: AsyncSession,
    catalogue: CatalogueClient,
    principal: Principal,
    project_id: uuid.UUID,
    data: ProjectUpdate,
) -> Project:
    """Edit: name proxied to the catalogue (manager-only); description local."""
    project = await require_project_access(session, catalogue, principal, project_id)
    fields = data.model_dump(exclude_unset=True)
    if fields.get("name") and fields["name"] != project.name:
        cat = await catalogue.update_project_name(
            str(project_id), fields["name"], principal.token
        )
        project.name = cat.get("name") or fields["name"]
    if "description" in fields:
        project.description = fields["description"]
    await session.commit()
    await session.refresh(project)
    return project


async def delete_project(
    session: AsyncSession,
    catalogue: CatalogueClient,
    principal: Principal,
    project_id: uuid.UUID,
) -> None:
    """Delete: proxied to the catalogue (manager-only), then soft-delete locally."""
    await require_project_access(session, catalogue, principal, project_id)
    await catalogue.delete_project(str(project_id), principal.token)
    await soft_delete_local(session, project_id)
