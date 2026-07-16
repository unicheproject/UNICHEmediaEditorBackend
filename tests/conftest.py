"""Pytest fixtures: SQLite async DB, eager job execution, httpx client.

Environment is configured BEFORE importing the app so the cached Settings and
the module-level async engine pick up the test database / storage / provider.

Auth + the catalogue are stubbed: ``get_current_principal`` returns a fixed test
principal and ``get_catalogue_client`` returns an in-memory ``FakeCatalogueClient``
so tests exercise the editor's logic without a running Keycloak/catalogue.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable

import pytest_asyncio

_TMP = tempfile.mkdtemp(prefix="uniche-test-")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP}/test.db"
os.environ["STORAGE_DIR"] = f"{_TMP}/storage"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["INFERENCE_PROVIDER"] = "mock"
os.environ["AGENT_PROVIDER"] = "mock"
os.environ["MAX_UPLOAD_SIZE_MB"] = "5"
os.environ["IDP_ISSUER_URI"] = "https://idp.test/realms/uniche"
os.environ["CATALOGUE_BASE_URL"] = "http://catalogue.test"

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.agent.executor import execute_plan  # noqa: E402
from app.api.v1.agent import get_plan_enqueuer  # noqa: E402
from app.api.v1.jobs import get_canceller, get_enqueuer  # noqa: E402
from app.core.database import async_session_factory, engine  # noqa: E402
from app.core.security import Principal, get_current_principal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Base  # noqa: E402
from app.services.catalogue_client import get_catalogue_client  # noqa: E402
from app.services.jobs import execute_job  # noqa: E402

# A fixed organisation the test principal "manages".
DEFAULT_ORG_ID = "11111111-1111-1111-1111-111111111111"
TEST_SUBJECT = "test-subject"


class FakeCatalogueClient:
    """In-memory stand-in for the catalogue: the test principal manages one org."""

    def __init__(self) -> None:
        self.projects: dict[str, dict] = {}

    def _store(self, *, project_id: str, org_id: str, name: str, slug: str) -> dict:
        record = {
            "id": project_id,
            "orgId": org_id,
            "name": name,
            "slug": slug,
            "status": "ACTIVE",
            "tool": {"slug": "media-editor", "name": "Media Editor"},
            "createdAt": "2026-06-30T00:00:00Z",
            "updatedAt": "2026-06-30T00:00:00Z",
        }
        self.projects[project_id] = record
        return record

    async def get_project(self, project_id: str, token: str) -> dict | None:
        return self.projects.get(project_id)

    async def list_authorization(self, token: str) -> dict:
        return {
            "subject": TEST_SUBJECT,
            "platformAdmin": False,
            "managedOrganisations": [DEFAULT_ORG_ID],
            "projectMemberships": [],
        }

    async def list_org_projects(self, org_id: str, token: str) -> list[dict]:
        return [p for p in self.projects.values() if p["orgId"] == org_id]

    async def list_organisations(self, token: str) -> list[dict]:
        return [{"id": DEFAULT_ORG_ID, "name": "Test Org", "slug": "test-org"}]

    async def create_project(
        self, org_id: str, name: str, slug: str, token: str
    ) -> dict:
        return self._store(
            project_id=str(uuid.uuid4()), org_id=org_id, name=name, slug=slug
        )

    async def update_project_name(
        self, project_id: str, name: str, token: str
    ) -> dict:
        record = self.projects[project_id]
        record["name"] = name
        return record

    async def delete_project(self, project_id: str, token: str) -> None:
        self.projects.pop(project_id, None)


async def _eager_enqueue(job_id: uuid.UUID) -> None:
    """Run the job synchronously in tests (no Redis/arq needed)."""
    async with async_session_factory() as session:
        await execute_job(session, job_id)


def _get_eager_enqueuer():
    return _eager_enqueue


async def _noop_cancel(job_id: uuid.UUID) -> bool:
    """No real arq/Redis in tests; services.jobs.cancel_job's DB write is
    what the cancel tests exercise."""
    return False


def _get_noop_canceller():
    return _noop_cancel


async def _eager_run_plan(plan_id: uuid.UUID) -> None:
    async with async_session_factory() as session:
        await execute_plan(session, plan_id)


def _get_eager_plan_enqueuer():
    return _eager_run_plan


@pytest_asyncio.fixture(autouse=True)
async def _reset_db() -> AsyncGenerator[None, None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    fake_catalogue = FakeCatalogueClient()
    app.dependency_overrides[get_enqueuer] = _get_eager_enqueuer
    app.dependency_overrides[get_canceller] = _get_noop_canceller
    app.dependency_overrides[get_plan_enqueuer] = _get_eager_plan_enqueuer
    app.dependency_overrides[get_catalogue_client] = lambda: fake_catalogue
    app.dependency_overrides[get_current_principal] = lambda: Principal(
        subject=TEST_SUBJECT, token="test-token", preferred_username="tester"
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
def make_project(
    client: AsyncClient,
) -> Callable[..., Awaitable[str]]:
    """Create a project (create-up via the fake catalogue) and return its id."""

    counter = {"n": 0}

    async def _make(name: str = "Project") -> str:
        counter["n"] += 1
        resp = await client.post(
            "/api/v1/projects",
            json={
                "name": name,
                "slug": f"proj-{counter['n']}-{uuid.uuid4().hex[:8]}",
                "org_id": DEFAULT_ORG_ID,
            },
        )
        assert resp.status_code == 201, resp.text
        return resp.json()["id"]

    return _make
