"""User-scoped Catalogue denial must not change shared project state."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.api.v1.jobs import get_enqueuer
from app.core.database import async_session_factory
from app.core.errors import NotFoundError
from app.core.security import Principal, get_current_principal
from app.main import app
from app.models import Base
from app.models.agent import AgentPlan
from app.models.project import Project
from app.services import projects as projects_svc
from app.services.catalogue_client import CatalogueClient, get_catalogue_client

API = "/api/v1"
TOMBSTONE = datetime(2026, 1, 1, tzinfo=UTC)


def _catalogue_client(
    monkeypatch: pytest.MonkeyPatch,
    respond: Callable[[httpx.Request], httpx.Response],
) -> CatalogueClient:
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        "app.services.catalogue_client.httpx.AsyncClient",
        lambda **kwargs: AsyncClient(transport=transport, **kwargs),
    )
    return CatalogueClient("http://catalogue.test", timeout=1)


async def _shared_state() -> dict[str, list[dict]]:
    """Read persisted rows, not an ORM identity map or an API response."""
    async with async_session_factory() as session:
        return {
            table.name: [dict(row) for row in (
                await session.execute(select(table).order_by(*table.primary_key.columns))
            ).mappings()]
            for table in Base.metadata.sorted_tables
        }


@pytest.mark.parametrize("deleted_at", [None, TOMBSTONE])
async def test_concealed_lookup_leaves_database_and_transaction_unchanged(
    client: AsyncClient, make_project, monkeypatch: pytest.MonkeyPatch, deleted_at
) -> None:
    project_id = uuid.UUID(await make_project("Owner's project"))
    async with async_session_factory() as session:
        project = await session.get(Project, project_id)
        project.description = "Owner's local description"
        project.deleted_at = deleted_at
        await session.commit()
    before = await _shared_state()
    requests: list[httpx.Request] = []

    def deny(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(404, json={"message": "Not found"})

    catalogue = _catalogue_client(monkeypatch, deny)
    principal = Principal(subject="user-b", token="user-b-test-token")
    async with async_session_factory() as session:
        commit = AsyncMock(wraps=session.commit)
        flush = AsyncMock(wraps=session.flush)
        monkeypatch.setattr(session, "commit", commit)
        monkeypatch.setattr(session, "flush", flush)
        assert not session.in_transaction()

        with pytest.raises(NotFoundError, match=f"Project '{project_id}' not found"):
            await projects_svc.require_project_access(session, catalogue, principal, project_id)

        commit.assert_not_called()
        flush.assert_not_called()
        assert not session.in_transaction()
        assert not session.new and not session.dirty and not session.deleted

    assert await _shared_state() == before
    assert len(requests) == 1
    assert requests[0].url.path == f"{API}/projects/{project_id}"
    assert requests[0].headers["Authorization"] == "Bearer user-b-test-token"


@pytest.mark.parametrize("operation", ["get", "patch", "delete", "composition"])
async def test_other_users_concealed_denial_cannot_mutate_owner_resources(
    client: AsyncClient, make_project, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    app.dependency_overrides[get_current_principal] = lambda: Principal(
        subject="user-a", token="user-a-test-token"
    )
    project_id = await make_project("User A's project")
    upload = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("source.mp4", b"lifecycle-test-media", "video/mp4")},
    )
    assert upload.status_code == 201, upload.text
    asset_id = upload.json()["id"]
    app.dependency_overrides[get_enqueuer] = lambda: AsyncMock()
    job = await client.post(f"{API}/jobs", json={
        "project_id": project_id,
        "asset_id": asset_id,
        "capability_id": "video.trim",
        "input": {"start": 0, "end": 1},
    })
    assert job.status_code == 201, job.text
    composition = await client.put(
        f"{API}/projects/{project_id}/composition",
        headers={"If-None-Match": "*"},
        json={
            "schema_version": 1,
            "canvas": {"width": 1920, "height": 1080},
            "audio_bed": None,
            "tracks": [{
                "id": str(uuid.uuid4()), "kind": "video",
                "clips": [{"id": str(uuid.uuid4()), "asset_id": asset_id}],
            }],
        },
    )
    assert composition.status_code == 201, composition.text
    agent = await client.post(
        f"{API}/agent/sessions", json={"project_id": project_id, "asset_ids": [asset_id]}
    )
    assert agent.status_code == 201, agent.text
    async with async_session_factory() as session:
        session.add(AgentPlan(
            session_id=uuid.UUID(agent.json()["id"]), summary="Owner's saved plan"
        ))
        await session.commit()
    before = await _shared_state()
    fake_catalogue = app.dependency_overrides[get_catalogue_client]()
    record = fake_catalogue.projects[project_id]

    def lookup(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == f"{API}/projects/{project_id}"
        if request.headers["Authorization"] == "Bearer user-a-test-token":
            return httpx.Response(200, json=record)
        assert request.headers["Authorization"] == "Bearer user-b-test-token"
        return httpx.Response(404, json={"message": "Not found"})

    catalogue = _catalogue_client(monkeypatch, lookup)
    app.dependency_overrides[get_catalogue_client] = lambda: catalogue
    app.dependency_overrides[get_current_principal] = lambda: Principal(
        subject="user-b", token="user-b-test-token"
    )
    path = f"{API}/projects/{project_id}"
    if operation == "composition":
        response = await client.get(f"{path}/composition")
    elif operation == "patch":
        response = await client.patch(path, json={"name": "Unauthorized rename"})
    else:
        response = await client.request(operation.upper(), path)

    assert response.status_code == 404, response.text
    assert response.json() == {
        "error": {"code": "not_found", "message": f"Project '{project_id}' not found"}
    }
    assert await _shared_state() == before


@pytest.mark.parametrize("deleted_at", [None, TOMBSTONE])
async def test_authorized_lookup_still_refreshes_and_restores_companion(
    client: AsyncClient, make_project, deleted_at
) -> None:
    project_id = await make_project("Old name")
    async with async_session_factory() as session:
        project = await session.get(Project, uuid.UUID(project_id))
        project.description = "Editor-local description"
        project.deleted_at = deleted_at
        await session.commit()
        created_at = project.created_at
    catalogue = app.dependency_overrides[get_catalogue_client]()
    new_org_id = str(uuid.uuid4())
    catalogue.projects[project_id].update(name="Canonical name", orgId=new_org_id)

    response = await client.get(f"{API}/projects/{project_id}")
    assert response.status_code == 200, response.text
    async with async_session_factory() as session:
        project = await session.get(Project, uuid.UUID(project_id))
        assert project.name == "Canonical name"
        assert project.org_id == uuid.UUID(new_org_id)
        assert project.deleted_at is None
        assert project.description == "Editor-local description"
        assert project.created_at == created_at


@pytest.mark.parametrize("delete_status", [204, 404])
async def test_local_deletion_requires_successful_explicit_catalogue_delete(
    client: AsyncClient, make_project, monkeypatch: pytest.MonkeyPatch, delete_status: int
) -> None:
    project_id = await make_project("Explicit deletion")
    fake_catalogue = app.dependency_overrides[get_catalogue_client]()
    record = fake_catalogue.projects[project_id]
    methods: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"{API}/projects/{project_id}"
        methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json=record)
        assert request.method == "DELETE"
        return httpx.Response(delete_status)

    catalogue = _catalogue_client(monkeypatch, respond)
    app.dependency_overrides[get_catalogue_client] = lambda: catalogue
    response = await client.delete(f"{API}/projects/{project_id}")
    assert response.status_code == delete_status, response.text
    assert methods == ["GET", "DELETE"]
    async with async_session_factory() as session:
        project = await session.get(Project, uuid.UUID(project_id))
        assert (project.deleted_at is not None) == (delete_status == 204)
        assert project.name == record["name"]
        assert project.org_id == uuid.UUID(record["orgId"])
