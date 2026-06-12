"""Pytest fixtures: SQLite async DB, eager job execution, httpx client.

Environment is configured BEFORE importing the app so the cached Settings and
the module-level async engine pick up the test database / storage / provider.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import AsyncGenerator

import pytest_asyncio

_TMP = tempfile.mkdtemp(prefix="uniche-test-")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP}/test.db"
os.environ["STORAGE_DIR"] = f"{_TMP}/storage"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["INFERENCE_PROVIDER"] = "mock"
os.environ["AGENT_PROVIDER"] = "mock"
os.environ["MAX_UPLOAD_SIZE_MB"] = "5"

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.agent.executor import execute_plan  # noqa: E402
from app.api.v1.agent import get_plan_enqueuer  # noqa: E402
from app.api.v1.jobs import get_enqueuer  # noqa: E402
from app.core.database import async_session_factory, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Base  # noqa: E402
from app.services.jobs import execute_job  # noqa: E402


async def _eager_enqueue(job_id: uuid.UUID) -> None:
    """Run the job synchronously in tests (no Redis/arq needed)."""
    async with async_session_factory() as session:
        await execute_job(session, job_id)


def _get_eager_enqueuer():
    return _eager_enqueue


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
    app.dependency_overrides[get_enqueuer] = _get_eager_enqueuer
    app.dependency_overrides[get_plan_enqueuer] = _get_eager_plan_enqueuer
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
