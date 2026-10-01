from __future__ import annotations

import io
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.core.database import async_session_factory
from app.models.agent import AgentPlan, AgentSession
from app.models.asset import Asset
from app.models.enums import AgentPlanStatus, JobStatus, MediaType, TechnicalMetadataStatus
from app.models.job import Job
from app.services import assets as assets_svc
from app.services.storage import get_storage
from app.services.temp_assets import cleanup_expired_temp_assets

API = "/api/v1"


async def _upload(client: AsyncClient, project_id: str) -> dict:
    response = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("source.png", io.BytesIO(b"small-test-image"), "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _mark_temp(asset_id: str, expires_at: datetime) -> None:
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(asset_id))
        assert asset is not None
        asset.is_intermediate = True
        asset.temp_expires_at = expires_at
        await session.commit()


async def test_temp_assets_are_hidden_by_default_and_include_intermediate_is_opt_in(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("temp-listing")
    original = await _upload(client, project_id)
    temp = await _upload(client, project_id)
    await _mark_temp(temp["id"], datetime.now(UTC) + timedelta(hours=72))

    normal = await client.get(f"{API}/projects/{project_id}/assets")
    included = await client.get(
        f"{API}/projects/{project_id}/assets?include_intermediate=true"
    )
    assert normal.status_code == included.status_code == 200
    assert {asset["id"] for asset in normal.json()} == {original["id"]}
    assert {asset["id"] for asset in included.json()} == {
        original["id"], temp["id"]
    }


async def test_derived_job_assets_remain_normal_unless_agent_marks_them_temp(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    project_id = await make_project("derived-normal")
    source = await _upload(client, project_id)
    output_file = tmp_path / "derived.png"
    output_file.write_bytes(b"derived-media")
    async with async_session_factory() as session:
        output = await assets_svc.create_derived_asset(
            session,
            project_id=uuid.UUID(project_id),
            source_asset_id=uuid.UUID(source["id"]),
            src_path=output_file,
            filename=output_file.name,
            media_type=MediaType.image,
        )
        assert output.is_intermediate is False
        assert output.temp_expires_at is None
    listed = await client.get(f"{API}/projects/{project_id}/assets")
    assert output.id.hex in {asset["id"].replace("-", "") for asset in listed.json()}


async def test_temp_asset_classification_and_retention_begin_at_persistence(
    client: AsyncClient, make_project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.media_metadata import TechnicalMetadata

    async def unavailable(*_args, **_kwargs):
        return TechnicalMetadata(status=TechnicalMetadataStatus.unavailable)

    monkeypatch.setattr(assets_svc, "probe_media_safely", unavailable)
    project_id = await make_project("temp-created")
    source = await _upload(client, project_id)
    output_file = tmp_path / "output.mp4"
    output_file.write_bytes(b"output-media")
    before = datetime.now(UTC)
    async with async_session_factory() as session:
        asset = await assets_svc.create_derived_asset(
            session,
            project_id=uuid.UUID(project_id),
            source_asset_id=uuid.UUID(source["id"]),
            src_path=output_file,
            filename=output_file.name,
            media_type=MediaType.video,
            is_intermediate=True,
        )
        after = datetime.now(UTC)
        assert asset.is_intermediate is True
        assert asset.temp_expires_at is not None
        # SQLite drops timezone metadata on round-tripping, so compare ISO
        # timestamps as naive UTC values there.
        expiry = asset.temp_expires_at.replace(tzinfo=None)
        assert before.replace(tzinfo=None) + timedelta(hours=72) <= expiry
        assert expiry <= after.replace(tzinfo=None) + timedelta(hours=72)


async def test_cleanup_respects_deadline_and_preserves_job_and_plan_audit(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("temp-expiry")
    original = await _upload(client, project_id)
    legacy = await _upload(client, project_id)
    uploaded = await _upload(client, project_id)
    now = datetime.now(UTC)
    await _mark_temp(uploaded["id"], now)
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(uploaded["id"]))
        assert asset is not None
        media_path = get_storage().get_path(asset.storage_path)
        original_asset = await session.get(Asset, uuid.UUID(original["id"]))
        assert original_asset is not None
        original_path = get_storage().get_path(original_asset.storage_path)
        legacy_asset = await session.get(Asset, uuid.UUID(legacy["id"]))
        assert legacy_asset is not None
        legacy_asset.is_intermediate = True
        legacy_path = get_storage().get_path(legacy_asset.storage_path)
        agent_session = AgentSession(project_id=uuid.UUID(project_id), asset_ids=[])
        session.add(agent_session)
        await session.flush()
        plan = AgentPlan(
            session_id=agent_session.id,
            status=AgentPlanStatus.failed,
            steps=[],
            step_runs=[{"output_asset_id": uploaded["id"]}],
            result_asset_ids=[],
        )
        job = Job(
            project_id=uuid.UUID(project_id),
            asset_id=uuid.UUID(uploaded["id"]),
            capability_id="video.resize",
            status=JobStatus.succeeded,
            input={},
            output={"outputs": [{"asset_id": uploaded["id"]}]},
        )
        session.add_all([plan, job])
        await session.commit()
        assert await cleanup_expired_temp_assets(session, now=now - timedelta(seconds=1)) == 0
        assert media_path.is_file()
        assert await cleanup_expired_temp_assets(session, now=now) == 1
        assert not media_path.exists()
        assert original_path.is_file()
        assert legacy_path.is_file()
        assert await cleanup_expired_temp_assets(session, now=now) == 0
        retained_asset = await session.get(Asset, uuid.UUID(uploaded["id"]))
        retained_job = await session.get(Job, job.id)
        retained_plan = await session.get(AgentPlan, plan.id)
        assert retained_asset is not None and retained_asset.physical_expired_at is not None
        assert retained_asset.deleted_at is None
        assert retained_job is not None
        assert retained_plan is not None
        assert retained_plan.step_runs[0]["output_asset_id"] == uploaded["id"]


async def test_active_job_dependency_prevents_temp_cleanup(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("temp-active-dependency")
    uploaded = await _upload(client, project_id)
    now = datetime.now(UTC)
    await _mark_temp(uploaded["id"], now - timedelta(hours=73))
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(uploaded["id"]))
        assert asset is not None
        path = get_storage().get_path(asset.storage_path)
        job = Job(
            project_id=uuid.UUID(project_id),
            asset_id=asset.id,
            capability_id="image.resize",
            status=JobStatus.queued,
            input={},
        )
        session.add(job)
        await session.commit()
        assert await cleanup_expired_temp_assets(session, now=now) == 0
        assert path.is_file()
        assert asset.physical_expired_at is None


async def test_approved_agent_plan_dependency_prevents_temp_cleanup(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("temp-active-plan")
    uploaded = await _upload(client, project_id)
    now = datetime.now(UTC)
    await _mark_temp(uploaded["id"], now - timedelta(hours=73))
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(uploaded["id"]))
        assert asset is not None
        path = get_storage().get_path(asset.storage_path)
        agent_session = AgentSession(
            project_id=uuid.UUID(project_id), asset_ids=[uploaded["id"]]
        )
        session.add(agent_session)
        await session.flush()
        plan = AgentPlan(
            session_id=agent_session.id,
            status=AgentPlanStatus.approved,
            steps=[{"asset": uploaded["id"]}],
            step_runs=[],
            result_asset_ids=[],
        )
        session.add(plan)
        await session.commit()
        assert await cleanup_expired_temp_assets(session, now=now) == 0
        assert path.is_file()
        assert asset.physical_expired_at is None


async def test_cleanup_marks_missing_binary_expired_and_keeps_technical_metadata(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("temp-missing-object")
    uploaded = await _upload(client, project_id)
    now = datetime.now(UTC)
    await _mark_temp(uploaded["id"], now - timedelta(hours=73))
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(uploaded["id"]))
        assert asset is not None
        asset.technical_metadata_status = TechnicalMetadataStatus.available.value
        asset.width = 640
        await session.commit()
        get_storage().delete_asset(asset.project_id, asset.id)
        assert await cleanup_expired_temp_assets(session, now=now) == 1
        assert asset.physical_expired_at is not None
        assert asset.width == 640
        assert asset.technical_metadata_status == TechnicalMetadataStatus.available.value


async def test_storage_failure_does_not_mark_asset_expired(
    client: AsyncClient, make_project, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id = await make_project("temp-delete-error")
    uploaded = await _upload(client, project_id)
    now = datetime.now(UTC)
    await _mark_temp(uploaded["id"], now - timedelta(hours=73))

    class FailingStorage:
        def delete_asset(self, *_args) -> None:
            raise PermissionError("private path deliberately omitted")

    monkeypatch.setattr("app.services.temp_assets.get_storage", lambda: FailingStorage())
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(uploaded["id"]))
        assert asset is not None
        path = get_storage().get_path(asset.storage_path)
        assert await cleanup_expired_temp_assets(session, now=now) == 0
        await session.refresh(asset)
        assert asset.physical_expired_at is None
        assert path.is_file()


async def test_expired_asset_metadata_remains_but_binary_endpoints_return_410(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("temp-expired-api")
    uploaded = await _upload(client, project_id)
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(uploaded["id"]))
        assert asset is not None
        asset.is_intermediate = True
        asset.temp_expires_at = datetime.now(UTC) - timedelta(hours=73)
        asset.physical_expired_at = datetime.now(UTC)
        asset.technical_metadata_status = TechnicalMetadataStatus.available.value
        asset.width = 24
        await session.commit()
        get_storage().delete_asset(asset.project_id, asset.id)

    metadata = await client.get(f"{API}/assets/{uploaded['id']}")
    assert metadata.status_code == 200
    assert metadata.json()["physical_expired_at"] is not None
    assert metadata.json()["width"] == 24
    assert metadata.json()["is_intermediate"] is True
    for endpoint in ("download", "thumbnail"):
        response = await client.get(f"{API}/assets/{uploaded['id']}/{endpoint}")
        assert response.status_code == 410
        assert response.json()["error"]["code"] == "asset_media_expired"
