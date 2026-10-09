"""Bounded, project-scoped batch soft deletion and per-item outcomes."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.dml import Update

from app.core.database import async_session_factory, get_session
from app.core.errors import NotFoundError
from app.core.security import get_current_principal
from app.main import app
from app.models.asset import Asset
from app.models.job import Job
from app.models.project import Project
from app.services.catalogue_client import get_catalogue_client
from app.services.storage import get_storage

API = "/api/v1"


async def _upload(client: AsyncClient, project_id: str) -> dict:
    response = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("source.png", b"batch-test-media", "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()


async def test_bulk_delete_hides_assets_but_retains_files_and_lineage(
    client: AsyncClient, make_project,
) -> None:
    project_id = await make_project()
    source = await _upload(client, project_id)
    derived = await _upload(client, project_id)
    survivor = await _upload(client, project_id)
    async with async_session_factory() as session:
        job = Job(
            project_id=uuid.UUID(project_id), asset_id=uuid.UUID(source["id"]),
            capability_id="image.resize", status="succeeded",
            output={"outputs": [{"asset_id": derived["id"]}]},
        )
        session.add(job)
        await session.flush()
        asset = await session.get(Asset, uuid.UUID(derived["id"]))
        asset.source_asset_id = uuid.UUID(source["id"])
        asset.source_job_id = job.id
        asset.provenance_kind = "derived"
        asset.ai_provenance = "inherited"
        await session.commit()
        job_id = job.id
        job_output = job.output

    ids = [derived["id"], source["id"]]
    response = await client.post(
        f"{API}/projects/{project_id}/assets/bulk-delete", json={"asset_ids": ids},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "results": [{"asset_id": asset_id, "status": "deleted"} for asset_id in ids],
    }
    listed = await client.get(f"{API}/projects/{project_id}/assets")
    assert [item["id"] for item in listed.json()] == [survivor["id"]]
    for asset_id in ids:
        for suffix in ("", "/download", "/thumbnail", "/waveform"):
            assert (await client.get(f"{API}/assets/{asset_id}{suffix}")).status_code == 404
        assert (await client.patch(
            f"{API}/assets/{asset_id}", json={"original_filename": "renamed.png"},
        )).status_code == 404
    async with async_session_factory() as session:
        source_row = await session.get(Asset, uuid.UUID(source["id"]))
        derived_row = await session.get(Asset, uuid.UUID(derived["id"]))
        assert source_row.deleted_at is not None
        assert derived_row.deleted_at == source_row.deleted_at
        assert derived_row.source_asset_id == source_row.id
        assert derived_row.source_job_id == job_id
        assert derived_row.provenance_kind == "derived"
        assert derived_row.ai_provenance == "inherited"
        assert (await session.get(Job, job_id)).output == job_output
        for uploaded in (source, derived):
            binary = get_storage().get_path(uploaded["storage_path"])
            assert binary.read_bytes() == b"batch-test-media"


async def test_bulk_delete_partial_results_conceal_foreign_missing_and_deleted_ids(
    client: AsyncClient, make_project,
) -> None:
    project_id = await make_project()
    other_project_id = await make_project()
    active = await _upload(client, project_id)
    deleted = await _upload(client, project_id)
    foreign = await _upload(client, other_project_id)
    assert (await client.delete(f"{API}/assets/{deleted['id']}")).status_code == 204
    async with async_session_factory() as session:
        original_deleted_at = (await session.get(Asset, uuid.UUID(deleted["id"]))).deleted_at
    missing = str(uuid.uuid4())
    ids = [foreign["id"], active["id"], missing, deleted["id"]]
    response = await client.post(
        f"{API}/projects/{project_id}/assets/bulk-delete", json={"asset_ids": ids},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"results": [
        {"asset_id": asset_id, "status": "deleted" if asset_id == active["id"] else "not_found"}
        for asset_id in ids
    ]}
    assert (await client.get(f"{API}/assets/{foreign['id']}")).status_code == 200
    async with async_session_factory() as session:
        deleted_row = await session.get(Asset, uuid.UUID(deleted["id"]))
        assert deleted_row.deleted_at == original_deleted_at


async def test_bulk_delete_repeated_request_is_safe(
    client: AsyncClient, make_project,
) -> None:
    project_id = await make_project()
    uploaded = await _upload(client, project_id)
    url = f"{API}/projects/{project_id}/assets/bulk-delete"
    payload = {"asset_ids": [uploaded["id"]]}
    assert (await client.post(url, json=payload)).json()["results"][0]["status"] == "deleted"
    async with async_session_factory() as session:
        deleted_at = (await session.get(Asset, uuid.UUID(uploaded["id"]))).deleted_at
    repeated = await client.post(url, json=payload)
    assert repeated.status_code == 200
    assert repeated.json() == {"results": [{"asset_id": uploaded["id"], "status": "not_found"}]}
    async with async_session_factory() as session:
        assert (await session.get(Asset, uuid.UUID(uploaded["id"]))).deleted_at == deleted_at


async def test_concurrent_bulk_deletions_report_actual_updates(
    client: AsyncClient, make_project,
) -> None:
    project_id = await make_project()
    uploaded = await _upload(client, project_id)
    url = f"{API}/projects/{project_id}/assets/bulk-delete"
    responses = await asyncio.gather(*[
        client.post(url, json={"asset_ids": [uploaded["id"]]}) for _ in range(2)
    ])
    assert all(response.status_code == 200 for response in responses)
    assert sorted(response.json()["results"][0]["status"] for response in responses) == [
        "deleted", "not_found",
    ]


@pytest.mark.parametrize("case", [
    "missing", "empty", "too_many", "duplicate", "bad_uuid", "null", "scalar", "extra",
])
async def test_bulk_delete_invalid_body_changes_no_assets(
    client: AsyncClient, make_project, case: str,
) -> None:
    project_id = await make_project()
    uploaded = await _upload(client, project_id)
    asset_id = uploaded["id"]
    bodies = {
        "missing": {}, "empty": {"asset_ids": []},
        "too_many": {"asset_ids": [asset_id] + [str(uuid.uuid4()) for _ in range(100)]},
        "duplicate": {"asset_ids": [asset_id, asset_id.upper()]},
        "bad_uuid": {"asset_ids": [asset_id, "not-a-uuid"]},
        "null": {"asset_ids": None}, "scalar": {"asset_ids": asset_id},
        "extra": {"asset_ids": [asset_id], "project_id": str(uuid.uuid4())},
    }
    response = await client.post(
        f"{API}/projects/{project_id}/assets/bulk-delete", json=bodies[case],
    )
    assert response.status_code == 422, response.text
    assert (await client.get(f"{API}/assets/{asset_id}")).status_code == 200


async def test_bulk_delete_accepts_maximum_batch(
    client: AsyncClient, make_project,
) -> None:
    project_id = await make_project()
    uploaded = await _upload(client, project_id)
    ids = [str(uuid.uuid4()) for _ in range(99)] + [uploaded["id"]]
    response = await client.post(
        f"{API}/projects/{project_id}/assets/bulk-delete", json={"asset_ids": ids},
    )
    assert response.status_code == 200
    assert response.json() == {"results": [
        {"asset_id": asset_id, "status": "deleted" if asset_id == uploaded["id"] else "not_found"}
        for asset_id in ids
    ]}


@pytest.mark.parametrize("denial", ["concealed", "unauthenticated"])
async def test_bulk_delete_denied_project_cannot_mutate_assets(
    client: AsyncClient, make_project, monkeypatch: pytest.MonkeyPatch, denial: str,
) -> None:
    project_id = await make_project()
    uploaded = await _upload(client, project_id)
    catalogue = app.dependency_overrides[get_catalogue_client]()
    lookup = AsyncMock(side_effect=NotFoundError("Project not found"))
    monkeypatch.setattr(catalogue, "get_project", lookup)
    if denial == "unauthenticated":
        app.dependency_overrides.pop(get_current_principal)
    response = await client.post(
        f"{API}/projects/{project_id}/assets/bulk-delete",
        json={"asset_ids": [uploaded["id"]]},
    )
    assert response.status_code == (401 if denial == "unauthenticated" else 404), response.text
    if denial == "unauthenticated":
        lookup.assert_not_called()
    else:
        lookup.assert_awaited_once()
    async with async_session_factory() as session:
        assert (await session.get(Asset, uuid.UUID(uploaded["id"]))).deleted_at is None
        assert (await session.get(Project, uuid.UUID(project_id))).deleted_at is None


async def test_bulk_delete_checks_only_named_project_authorization(
    client: AsyncClient, make_project, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = await make_project()
    foreign = await _upload(client, await make_project())
    catalogue = app.dependency_overrides[get_catalogue_client]()
    lookup = AsyncMock(wraps=catalogue.get_project)
    monkeypatch.setattr(catalogue, "get_project", lookup)
    response = await client.post(
        f"{API}/projects/{project_id}/assets/bulk-delete",
        json={"asset_ids": [foreign["id"]]},
    )
    assert response.status_code == 200
    assert response.json() == {"results": [{"asset_id": foreign["id"], "status": "not_found"}]}
    lookup.assert_awaited_once()
    assert lookup.await_args.args[0] == project_id


async def test_bulk_delete_intermediate_lifecycle_is_unchanged(
    client: AsyncClient, make_project,
) -> None:
    project_id = await make_project()
    uploaded = await _upload(client, project_id)
    expires_at = datetime(2026, 1, 1, tzinfo=UTC)
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(uploaded["id"]))
        asset.is_intermediate = True
        asset.temp_expires_at = expires_at
        asset.physical_expired_at = expires_at
        await session.commit()
    response = await client.post(
        f"{API}/projects/{project_id}/assets/bulk-delete",
        json={"asset_ids": [uploaded["id"]]},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "deleted"
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(uploaded["id"]))
        assert asset.deleted_at is not None
        assert asset.is_intermediate
        assert asset.temp_expires_at.replace(tzinfo=UTC) == expires_at
        assert asset.physical_expired_at.replace(tzinfo=UTC) == expires_at


@pytest.mark.parametrize("failure", ["update", "commit"])
async def test_bulk_delete_database_failure_rolls_back_entire_batch(
    client: AsyncClient, make_project, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    project_id = await make_project()
    uploaded = [await _upload(client, project_id) for _ in range(2)]

    async def failing_session():
        async with async_session_factory() as session:
            real_commit = session.commit
            real_execute = session.execute

            async def fail_bulk_update(statement, *args, **kwargs):
                if isinstance(statement, Update) and statement.table.name == "assets":
                    raise SQLAlchemyError("synthetic-private-database-detail")
                return await real_execute(statement, *args, **kwargs)

            async def fail_bulk_commit():
                # Authorization may refresh the project first; fail only the
                # transaction that actually deleted Assets.
                deleted = await session.scalar(
                    select(Asset.id).where(Asset.deleted_at.is_not(None))
                )
                if deleted is not None:
                    raise SQLAlchemyError("synthetic-private-database-detail")
                await real_commit()

            if failure == "commit":
                monkeypatch.setattr(session, "commit", fail_bulk_commit)
            else:
                monkeypatch.setattr(session, "execute", fail_bulk_update)
            yield session

    app.dependency_overrides[get_session] = failing_session
    response = await client.post(
        f"{API}/projects/{project_id}/assets/bulk-delete",
        json={"asset_ids": [item["id"] for item in uploaded]},
    )
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "Assets could not be deleted."},
    }
    async with async_session_factory() as session:
        for item in uploaded:
            assert (await session.get(Asset, uuid.UUID(item["id"]))).deleted_at is None


def test_bulk_delete_openapi_contract() -> None:
    schema = app.openapi()
    operation = schema["paths"][f"{API}/projects/{{project_id}}/assets/bulk-delete"]["post"]
    assert operation["requestBody"]["required"]
    assert "200" in operation["responses"]
    definitions = schema["components"]["schemas"]
    ids = definitions["AssetBulkDeleteRequest"]["properties"]["asset_ids"]
    assert ids["minItems"] == 1
    assert ids["maxItems"] == 100
    assert ids["uniqueItems"] is True
    assert ids["items"]["format"] == "uuid"
    assert definitions["AssetBulkDeleteItem"]["properties"]["status"]["enum"] == [
        "deleted", "not_found",
    ]
