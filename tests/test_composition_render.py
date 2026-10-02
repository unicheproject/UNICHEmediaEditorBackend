from __future__ import annotations

import io
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select

from app.capabilities import registry
from app.capabilities.context import JobContext
from app.composition.render import (
    COMPOSITION_RENDER_COMPILER_VERSION,
    compile_composition_revision,
    compose_request_from_job_context,
)
from app.core.database import async_session_factory
from app.core.errors import NotFoundError, ValidationError
from app.models.asset import Asset
from app.models.composition import ProjectCompositionRevision
from app.models.job import Job
from app.services.composition_render import resolve_composition_render
from app.tools import compose_renderer

API = "/api/v1"


def _document(video_ids: list[str], *, audio_id: str | None = None) -> dict:
    return {
        "schema_version": 1,
        "canvas": {"width": 1280, "height": 720},
        "tracks": [
            {
                "id": str(uuid.uuid4()),
                "kind": "video",
                "clips": [
                    {"id": str(uuid.uuid4()), "asset_id": asset_id}
                    for asset_id in video_ids
                ],
            }
        ],
        "audio_bed": (
            {"asset_id": audio_id, "mode": "duck", "volume": 0.17}
            if audio_id
            else None
        ),
    }


async def _upload(client: AsyncClient, project_id: str, name: str) -> str:
    response = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": (name, io.BytesIO(b"render-spec-fixture"), "application/octet-stream")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def _save(client: AsyncClient, project_id: str, document: dict) -> dict:
    response = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=document,
        headers={"If-None-Match": "*"},
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _compiled(revision_id: str):
    async with async_session_factory() as session:
        revision = await session.get(ProjectCompositionRevision, uuid.UUID(revision_id))
        assert revision is not None
        return compile_composition_revision(revision)


async def test_compile_preserves_order_duplicates_canvas_audio_and_occurrences(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project()
    first = await _upload(client, project_id, "first.mp4")
    second = await _upload(client, project_id, "second.mp4")
    audio = await _upload(client, project_id, "bed.mp3")
    document = _document([first, second, first], audio_id=audio)
    clip_ids = [clip["id"] for clip in document["tracks"][0]["clips"]]
    saved = await _save(client, project_id, document)

    async with async_session_factory() as session:
        job_count_before = await session.scalar(select(func.count()).select_from(Job))
        asset_count_before = await session.scalar(select(func.count()).select_from(Asset))
        revision = await session.get(
            ProjectCompositionRevision, uuid.UUID(saved["revision_id"])
        )
        assert revision is not None
        compiled = compile_composition_revision(revision)
        job_count_after = await session.scalar(select(func.count()).select_from(Job))
        asset_count_after = await session.scalar(select(func.count()).select_from(Asset))

    request = compiled.request
    assert request.video_asset_ids == tuple(map(uuid.UUID, [first, second, first]))
    assert (request.width, request.height) == (1280, 720)
    assert request.audio_asset_id == uuid.UUID(audio)
    assert request.audio_mode.value == "duck"
    assert request.music_volume == 0.17
    assert [str(item.clip_id) for item in compiled.source.clip_occurrences] == clip_ids
    assert [item.asset_id for item in compiled.source.clip_occurrences] == list(
        request.video_asset_ids
    )
    assert compiled.source.revision_id == uuid.UUID(saved["revision_id"])
    assert compiled.source.revision_number == 1
    assert compiled.source.compiler_version == COMPOSITION_RENDER_COMPILER_VERSION
    assert job_count_after == job_count_before
    assert asset_count_after == asset_count_before


async def test_compiler_targets_immutable_revision_after_head_advances(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project()
    video = await _upload(client, project_id, "clip.mp4")
    first = await _save(client, project_id, _document([video]))
    first_compiled = await _compiled(first["revision_id"])
    assert first_compiled.request.audio_asset_id is None
    assert first_compiled.request.audio_mode.value == "mix"
    assert first_compiled.request.music_volume == 0.3

    changed = first_compiled.request
    update_doc = _document([video])
    update_doc["canvas"] = {"width": 640, "height": 360}
    response = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=update_doc,
        headers={"If-Match": (await client.get(
            f"{API}/projects/{project_id}/composition"
        )).headers["etag"]},
    )
    assert response.status_code == 200
    second = await _compiled(response.json()["revision_id"])
    first_again = await _compiled(first["revision_id"])

    assert first_again.request == first_compiled.request
    assert first_again.source == first_compiled.source
    assert second.request.width == 640
    assert second.request != changed


@pytest.mark.parametrize(
    "document_change",
    [
        lambda document: document.update(schema_version=2),
        lambda document: document.update(unexpected="value"),
    ],
)
async def test_compiler_rejects_corrupt_persisted_documents(document_change) -> None:
    document = _document([str(uuid.uuid4())])
    document_change(document)
    fake_revision = SimpleNamespace(
        id=uuid.uuid4(),
        composition_id=uuid.uuid4(),
        revision_number=1,
        document=document,
        document_hash="0" * 64,
    )
    with pytest.raises(ValidationError, match="Saved composition revision is invalid"):
        compile_composition_revision(fake_revision)


async def test_compiler_rejects_document_hash_mismatch() -> None:
    document = _document([str(uuid.uuid4())])
    fake_revision = SimpleNamespace(
        id=uuid.uuid4(),
        composition_id=uuid.uuid4(),
        revision_number=1,
        document=document,
        document_hash="0" * 64,
    )
    with pytest.raises(ValidationError, match="integrity check"):
        compile_composition_revision(fake_revision)


async def test_direct_compose_and_compiled_spec_use_shared_renderer(
    client: AsyncClient, make_project, monkeypatch, tmp_path
) -> None:
    project_id = await make_project()
    first = await _upload(client, project_id, "first.mp4")
    second = await _upload(client, project_id, "second.mp4")
    audio = await _upload(client, project_id, "bed.mp3")
    document = _document([first, second, first], audio_id=audio)
    saved = await _save(client, project_id, document)
    compiled = await _compiled(saved["revision_id"])

    direct_context = JobContext(
        capability_id="video.compose",
        params={
            "asset_ids": [first, second, first],
            "width": 1280,
            "height": 720,
            "audio_asset_id": audio,
            "audio_mode": "duck",
            "music_volume": 0.17,
        },
        input_paths=["/tmp/first", "/tmp/second", "/tmp/first"],
        input_asset_ids=list(map(uuid.UUID, [first, second, first])),
        named_input_paths={"audio_asset_id": "/tmp/bed"},
        work_dir=tmp_path,
    )
    direct_request = compose_request_from_job_context(direct_context)
    assert direct_request == compiled.request

    calls: list[tuple] = []

    async def fake_ffmpeg_compose(segments, dst, **kwargs):
        calls.append((segments, dst, kwargs))

    monkeypatch.setattr(compose_renderer.ffmpeg, "compose", fake_ffmpeg_compose)
    await compose_renderer.render_compose(
        compiled.request,
        compose_renderer.ComposeRenderSources(
            video_paths=("/tmp/first", "/tmp/second", "/tmp/first"),
            audio_path="/tmp/bed",
        ),
        "/tmp/output.mp4",
        work_dir=tmp_path,
    )
    assert calls == [
        (
            ["/tmp/first", "/tmp/second", "/tmp/first"],
            "/tmp/output.mp4",
            {
                "width": 1280,
                "height": 720,
                "work_dir": tmp_path,
                "audio_path": "/tmp/bed",
                "audio_mode": "duck",
                "music_volume": 0.17,
                "subtitle_path": None,
            },
        )
    ]


@pytest.mark.parametrize(
    "invalid_state",
    [
        "wrong_project",
        "wrong_media_type",
        "wrong_audio_type",
        "deleted",
        "temporary",
        "expired",
        "missing_file",
        "missing_row",
    ],
)
async def test_runtime_resolver_revalidates_asset_state(
    client: AsyncClient, make_project, invalid_state: str
) -> None:
    project_id = await make_project()
    other_project_id = await make_project("Other")
    video = await _upload(client, project_id, "clip.mp4")
    audio = (
        await _upload(client, project_id, "bed.mp3")
        if invalid_state == "wrong_audio_type"
        else None
    )
    saved = await _save(client, project_id, _document([video], audio_id=audio))
    async with async_session_factory() as session:
        asset_id = audio if invalid_state == "wrong_audio_type" else video
        assert asset_id is not None
        asset = await session.get(Asset, uuid.UUID(asset_id))
        assert asset is not None
        if invalid_state == "wrong_project":
            asset.project_id = uuid.UUID(other_project_id)
        elif invalid_state in {"wrong_media_type", "wrong_audio_type"}:
            from app.models.enums import MediaType

            asset.media_type = (
                MediaType.audio
                if invalid_state == "wrong_media_type"
                else MediaType.video
            )
        elif invalid_state == "deleted":
            asset.deleted_at = datetime.now(UTC)
        elif invalid_state == "temporary":
            asset.is_intermediate = True
            asset.temp_expires_at = datetime.now(UTC)
        elif invalid_state == "expired":
            asset.physical_expired_at = datetime.now(UTC)
        elif invalid_state == "missing_file":
            from app.services.storage import get_storage

            get_storage().get_path(asset.storage_path).unlink()
        elif invalid_state == "missing_row":
            await session.delete(asset)
        await session.commit()

    async with async_session_factory() as session:
        with pytest.raises(ValidationError) as raised:
            await resolve_composition_render(
                session, uuid.UUID(project_id), uuid.UUID(saved["revision_id"])
            )
    assert raised.value.code == "composition_asset_unavailable"


async def test_runtime_resolver_returns_ordered_project_owned_sources(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project()
    first = await _upload(client, project_id, "first.mp4")
    second = await _upload(client, project_id, "second.mp4")
    audio = await _upload(client, project_id, "bed.mp3")
    saved = await _save(
        client, project_id, _document([first, second, first], audio_id=audio)
    )
    async with async_session_factory() as session:
        resolved = await resolve_composition_render(
            session, uuid.UUID(project_id), uuid.UUID(saved["revision_id"])
        )
    assert resolved.compiled.request.video_asset_ids == tuple(
        map(uuid.UUID, [first, second, first])
    )
    assert resolved.sources.video_paths[0] == resolved.sources.video_paths[2]
    assert resolved.sources.video_paths[0] != resolved.sources.video_paths[1]
    assert resolved.sources.audio_path is not None


def test_video_compose_capability_contract_remains_unchanged() -> None:
    capability = registry.get("video.compose")
    assert set(capability.input_schema["properties"]) == {
        "asset_ids",
        "width",
        "height",
        "audio_asset_id",
        "audio_mode",
        "music_volume",
        "subtitle_asset_id",
    }
    assert not registry.exists("project.export")


async def test_no_public_composition_render_route_or_export_capability(
    client: AsyncClient,
) -> None:
    openapi = (await client.get("/openapi.json")).json()
    paths = openapi["paths"]
    assert "/api/v1/projects/{project_id}/composition/render" not in paths
    assert not registry.exists("project.export")


async def test_runtime_resolver_conceals_revision_from_another_project(
    client: AsyncClient, make_project
) -> None:
    project_a = await make_project()
    project_b = await make_project("Other")
    video = await _upload(client, project_a, "clip.mp4")
    saved = await _save(client, project_a, _document([video]))
    async with async_session_factory() as session:
        with pytest.raises(NotFoundError) as raised:
            await resolve_composition_render(
                session, uuid.UUID(project_b), uuid.UUID(saved["revision_id"])
            )
    assert getattr(raised.value, "code", None) == "composition_revision_not_found"
