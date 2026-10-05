"""Unscoped historical Jobs must not bypass project authorization."""

import io
import uuid
from unittest.mock import AsyncMock, Mock

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.api.v1 import jobs as jobs_api
from app.core.database import async_session_factory
from app.main import app
from app.models.enums import JobStatus
from app.models.job import Job
from app.services.catalogue_client import get_catalogue_client

API = "/api/v1"


async def _save_job(
    *,
    job_id: uuid.UUID | None = None,
    project_id: str | None = None,
    asset_id: str | None = None,
    status: JobStatus = JobStatus.queued,
) -> uuid.UUID:
    async with async_session_factory() as session:
        job = Job(
            id=job_id or uuid.uuid4(),
            project_id=uuid.UUID(project_id) if project_id else None,
            asset_id=uuid.UUID(asset_id) if asset_id else None,
            capability_id="image.resize",
            status=status,
            input={"private": "legacy input must not leak"},
            output={"private": "legacy output must not leak"},
            progress=17,
        )
        session.add(job)
        await session.commit()
        return job.id


async def _snapshot(job_id: uuid.UUID) -> dict:
    async with async_session_factory() as session:
        row = (
            await session.execute(select(Job.__table__).where(Job.id == job_id))
        ).mappings().one()
        return dict(row)


@pytest.mark.parametrize(
    "method,suffix", [("GET", ""), ("POST", "/cancel"), ("GET", "/events")]
)
@pytest.mark.parametrize("status", [JobStatus.queued, JobStatus.running])
async def test_unscoped_job_is_concealed_without_side_effects(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    suffix: str,
    status: JobStatus,
) -> None:
    canceller = AsyncMock(return_value=True)
    app.dependency_overrides[jobs_api.get_canceller] = lambda: canceller
    cancel_service = AsyncMock(wraps=jobs_api.svc.cancel_job)
    monkeypatch.setattr(jobs_api.svc, "cancel_job", cancel_service)
    stream = Mock(side_effect=AssertionError("Unauthorized stream opened"))
    polling_session = Mock(side_effect=AssertionError("Unauthorized progress polling"))
    monkeypatch.setattr(jobs_api, "EventSourceResponse", stream)
    monkeypatch.setattr(jobs_api, "async_session_factory", polling_session)

    job_id = uuid.uuid4()
    path = f"{API}/jobs/{job_id}{suffix}"
    missing = await client.request(method, path)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"

    await _save_job(job_id=job_id, status=status)
    before = await _snapshot(job_id)
    response = await client.request(method, path)

    assert response.status_code == 404
    assert response.json() == missing.json()
    assert await _snapshot(job_id) == before
    cancel_service.assert_not_awaited()
    canceller.assert_not_awaited()
    stream.assert_not_called()
    polling_session.assert_not_called()


async def test_primary_asset_does_not_authorize_an_unscoped_job(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("Accessible source, unscoped Job")
    upload = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("source.png", io.BytesIO(b"image"), "image/png")},
    )
    assert upload.status_code == 201
    job_id = await _save_job(asset_id=upload.json()["id"])

    response = await client.get(f"{API}/jobs/{job_id}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert (await _snapshot(job_id))["project_id"] is None


async def test_authorized_scoped_job_detail_and_cancellation_still_work(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("Scoped Job")
    job_id = await _save_job(project_id=project_id)
    canceller = AsyncMock(return_value=True)
    app.dependency_overrides[jobs_api.get_canceller] = lambda: canceller

    detail = await client.get(f"{API}/jobs/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["project_id"] == project_id
    cancelled = await client.post(f"{API}/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert (await _snapshot(job_id))["status"] == JobStatus.cancelled
    canceller.assert_awaited_once_with(job_id)


async def test_denied_scoped_job_stays_concealed_and_unchanged(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("Denied scoped Job")
    job_id = await _save_job(project_id=project_id)
    before = await _snapshot(job_id)
    catalogue = app.dependency_overrides[get_catalogue_client]()
    catalogue.projects.pop(project_id)

    response = await client.get(f"{API}/jobs/{job_id}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert await _snapshot(job_id) == before


async def test_project_history_excludes_unscoped_jobs(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("Project history")
    scoped_id = await _save_job(project_id=project_id)
    unscoped_id = await _save_job()

    response = await client.get(f"{API}/projects/{project_id}/jobs")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert [item["id"] for item in response.json()["items"]] == [str(scoped_id)]
    assert str(unscoped_id) not in response.text


@pytest.mark.parametrize("explicit_project", [True, False])
async def test_normal_submission_persists_authorized_project_scope(
    client: AsyncClient, make_project, explicit_project: bool
) -> None:
    project_id = await make_project("Job producer scope")
    upload = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("source.png", io.BytesIO(b"image"), "image/png")},
    )
    assert upload.status_code == 201
    enqueuer = AsyncMock()
    app.dependency_overrides[jobs_api.get_enqueuer] = lambda: enqueuer
    body = {
        "capability_id": "image.resize",
        "asset_id": upload.json()["id"],
        "input": {"width": 32, "height": 24},
    }
    if explicit_project:
        body["project_id"] = project_id

    response = await client.post(f"{API}/jobs", json=body)

    assert response.status_code == 201, response.text
    job_id = uuid.UUID(response.json()["id"])
    assert response.json()["project_id"] == project_id
    assert (await _snapshot(job_id))["project_id"] == uuid.UUID(project_id)
    enqueuer.assert_awaited_once_with(job_id)
