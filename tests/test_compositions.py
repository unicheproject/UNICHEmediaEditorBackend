from __future__ import annotations

import asyncio
import io
import uuid
from datetime import UTC, datetime

import pytest
from httpx import AsyncClient
from pydantic import ValidationError as PydanticValidationError

from app.core.database import async_session_factory
from app.main import app
from app.models.asset import Asset
from app.models.composition import ProjectCompositionRevision
from app.schemas.composition import CompositionDocument
from app.services.compositions import canonical_document
from app.services.storage import get_storage

API = "/api/v1"


def _clip(asset_id: str, clip_id: str | None = None) -> dict:
    return {"id": clip_id or str(uuid.uuid4()), "asset_id": asset_id}


def _document(video_ids: list[str], *, audio_id: str | None = None) -> dict:
    return {
        "schema_version": 1,
        "canvas": {"width": 1920, "height": 1080},
        "tracks": [
            {
                "id": str(uuid.uuid4()),
                "kind": "video",
                "clips": [_clip(asset_id) for asset_id in video_ids],
            }
        ],
        "audio_bed": (
            {"asset_id": audio_id, "mode": "mix", "volume": 0.3}
            if audio_id
            else None
        ),
    }


async def _upload(client: AsyncClient, project_id: str, filename: str) -> str:
    response = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={
            "file": (
                filename,
                io.BytesIO(b"composition-test-media"),
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_schema_v1_defaults_and_repeated_asset_occurrences() -> None:
    asset_id = str(uuid.uuid4())
    doc = _document([asset_id, asset_id])
    doc["tracks"][0]["clips"][0]["id"] = str(uuid.uuid4())
    doc["tracks"][0]["clips"][1]["id"] = str(uuid.uuid4())
    parsed = CompositionDocument.model_validate(doc)
    assert len(parsed.tracks[0].clips) == 2
    assert parsed.audio_bed is None


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version=2),
        lambda d: d["canvas"].update(width=0),
        lambda d: d["canvas"].update(height=9000),
        lambda d: d["canvas"].update(width=1919),
        lambda d: d["tracks"].clear(),
        lambda d: d["tracks"].append(d["tracks"][0]),
        lambda d: d["tracks"][0].update(kind="audio"),
        lambda d: d["tracks"][0]["clips"].append(d["tracks"][0]["clips"][0]),
        lambda d: d.update(unrecognized=True),
    ],
)
def test_schema_rejects_invalid_v1_documents(change) -> None:
    document = _document([str(uuid.uuid4())])
    change(document)
    with pytest.raises(PydanticValidationError):
        CompositionDocument.model_validate(document)


@pytest.mark.parametrize("mode", ["unknown", "", "MIX"])
def test_schema_rejects_unknown_audio_modes(mode: str) -> None:
    document = _document([str(uuid.uuid4())], audio_id=str(uuid.uuid4()))
    document["audio_bed"]["mode"] = mode
    with pytest.raises(PydanticValidationError):
        CompositionDocument.model_validate(document)


@pytest.mark.parametrize("volume", [-0.1, 1.01, float("nan"), float("inf")])
def test_schema_rejects_invalid_audio_volume(volume: float) -> None:
    document = _document([str(uuid.uuid4())], audio_id=str(uuid.uuid4()))
    document["audio_bed"]["volume"] = volume
    with pytest.raises(PydanticValidationError):
        CompositionDocument.model_validate(document)


def test_canonical_hash_ignores_object_key_order_but_preserves_clip_order() -> None:
    first = _document([str(uuid.uuid4()), str(uuid.uuid4())])
    parsed = CompositionDocument.model_validate(first)
    reversed_keys = {
        "audio_bed": None,
        "tracks": first["tracks"],
        "canvas": first["canvas"],
        "schema_version": 1,
    }
    _, hash_a = canonical_document(parsed)
    _, hash_b = canonical_document(CompositionDocument.model_validate(reversed_keys))
    assert hash_a == hash_b

    reversed_keys["tracks"] = [
        {**first["tracks"][0], "clips": list(reversed(first["tracks"][0]["clips"]))}
    ]
    _, hash_c = canonical_document(CompositionDocument.model_validate(reversed_keys))
    assert hash_c != hash_a


async def test_first_save_get_revision_noop_and_immutable_update(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project()
    video_a = await _upload(client, project_id, "a.mp4")
    video_b = await _upload(client, project_id, "b.mp4")
    audio = await _upload(client, project_id, "bed.mp3")
    original = _document([video_a, video_a, video_b], audio_id=audio)

    missing = await client.get(f"{API}/projects/{project_id}/composition")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "composition_not_found"

    no_precondition = await client.put(
        f"{API}/projects/{project_id}/composition", json=original
    )
    assert no_precondition.status_code == 428

    created = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=original,
        headers={"If-None-Match": "*"},
    )
    assert created.status_code == 201, created.text
    first = created.json()
    first_etag = created.headers["etag"]
    assert first["revision_number"] == 1
    assert first["document"]["audio_bed"] == {
        "asset_id": audio,
        "mode": "mix",
        "volume": 0.3,
    }
    async with async_session_factory() as session:
        revision_row = await session.get(
            ProjectCompositionRevision, uuid.UUID(first["revision_id"])
        )
        assert revision_row is not None
        assert revision_row.author_subject == "test-subject"

    current = await client.get(
        f"{API}/projects/{project_id}/composition",
        headers={"Origin": "http://localhost:5174"},
    )
    assert current.status_code == 200
    assert current.headers["etag"] == first_etag
    assert "etag" in current.headers["access-control-expose-headers"].lower()
    assert current.json() == first

    historical = await client.get(
        f"{API}/projects/{project_id}/composition/revisions/{first['revision_id']}"
    )
    assert historical.status_code == 200
    assert historical.json() == first

    # Different input object-key order has the same normalized content.
    equivalent = {
        "audio_bed": original["audio_bed"],
        "tracks": original["tracks"],
        "canvas": original["canvas"],
        "schema_version": 1,
    }
    noop = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=equivalent,
        headers={"If-Match": first_etag},
    )
    assert noop.status_code == 200
    assert noop.json()["revision_id"] == first["revision_id"]
    assert noop.headers["etag"] == first_etag

    changed = {**original, "canvas": {"width": 1280, "height": 720}}
    updated = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=changed,
        headers={"If-Match": first_etag},
    )
    assert updated.status_code == 200, updated.text
    second = updated.json()
    assert second["revision_number"] == 2
    assert second["revision_id"] != first["revision_id"]
    assert second["document"]["canvas"] == changed["canvas"]
    assert (await client.get(
        f"{API}/projects/{project_id}/composition/revisions/{first['revision_id']}"
    )).json() == first

    stale = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=changed,
        headers={"If-Match": first_etag},
    )
    assert stale.status_code == 412
    assert stale.json()["error"]["code"] == "precondition_failed"
    assert (await client.get(f"{API}/projects/{project_id}/composition")).json() == second


async def test_composition_rejects_missing_wrong_project_deleted_temp_and_expired_assets(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project()
    other_project_id = await make_project("Other Project")
    video = await _upload(client, project_id, "clip.mp4")
    audio = await _upload(client, project_id, "bed.mp3")
    image = await _upload(client, project_id, "still.png")
    other_video = await _upload(client, other_project_id, "other.mp4")

    valid = _document([video], audio_id=audio)
    saved = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=valid,
        headers={"If-None-Match": "*"},
    )
    assert saved.status_code == 201

    cases = [
        _document([str(uuid.uuid4())]),
        _document([audio]),
        _document([image]),
        _document([other_video]),
        _document([video], audio_id=image),
    ]
    for invalid in cases:
        response = await client.put(
            f"{API}/projects/{project_id}/composition",
            json=invalid,
            headers={"If-Match": saved.headers["etag"]},
        )
        assert response.status_code == 422, response.text

    await client.delete(f"{API}/assets/{video}")
    deleted = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=_document([video]),
        headers={"If-Match": saved.headers["etag"]},
    )
    assert deleted.status_code == 422
    historical = await client.get(
        f"{API}/projects/{project_id}/composition/revisions/{saved.json()['revision_id']}"
    )
    assert historical.status_code == 200
    assert historical.json()["revision_id"] == saved.json()["revision_id"]

    fresh = await _upload(client, project_id, "fresh.mp4")
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(fresh))
        assert asset is not None
        asset.is_intermediate = True
        asset.temp_expires_at = datetime.now(UTC)
        await session.commit()
    temp = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=_document([fresh]),
        headers={"If-Match": saved.headers["etag"]},
    )
    assert temp.status_code == 422

    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(fresh))
        assert asset is not None
        asset.is_intermediate = False
        asset.temp_expires_at = None
        asset.physical_expired_at = datetime.now(UTC)
        await session.commit()
    expired = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=_document([fresh]),
        headers={"If-Match": saved.headers["etag"]},
    )
    assert expired.status_code == 422


async def test_composition_rejects_missing_binary_without_creating_revision(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project()
    video_id = await _upload(client, project_id, "clip.mp4")
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(video_id))
        assert asset is not None
        get_storage().get_path(asset.storage_path).unlink()

    failed = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=_document([video_id]),
        headers={"If-None-Match": "*"},
    )
    assert failed.status_code == 422
    assert (await client.get(f"{API}/projects/{project_id}/composition")).status_code == 404


async def test_composition_preconditions_and_authorization(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project()
    video_id = await _upload(client, project_id, "clip.mp4")
    document = _document([video_id])

    created = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=document,
        headers={"If-None-Match": "*"},
    )
    assert created.status_code == 201
    duplicate_first = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=document,
        headers={"If-None-Match": "*"},
    )
    assert duplicate_first.status_code == 412

    missing_update_precondition = await client.put(
        f"{API}/projects/{project_id}/composition", json=document
    )
    assert missing_update_precondition.status_code == 428
    malformed = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=document,
        headers={"If-Match": "W/\"weak\""},
    )
    assert malformed.status_code == 400
    both = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=document,
        headers={"If-Match": created.headers["etag"], "If-None-Match": "*"},
    )
    assert both.status_code == 400

    # Removing the project from the Catalogue fake denies access even though
    # the caller still knows the project and revision UUIDs.
    from app.services.catalogue_client import get_catalogue_client

    fake_catalogue = app.dependency_overrides[get_catalogue_client]()
    fake_catalogue.projects.pop(project_id)
    denied_current = await client.get(f"{API}/projects/{project_id}/composition")
    denied_revision = await client.get(
        f"{API}/projects/{project_id}/composition/revisions/{created.json()['revision_id']}"
    )
    assert denied_current.status_code == 404
    assert denied_revision.status_code == 404
    denied_save = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=document,
        headers={"If-Match": created.headers["etag"]},
    )
    assert denied_save.status_code == 404


async def test_revision_from_another_project_is_concealed(
    client: AsyncClient, make_project
) -> None:
    project_a = await make_project("Project A")
    project_b = await make_project("Project B")
    video_a = await _upload(client, project_a, "a.mp4")
    video_b = await _upload(client, project_b, "b.mp4")
    created_a = await client.put(
        f"{API}/projects/{project_a}/composition",
        json=_document([video_a]),
        headers={"If-None-Match": "*"},
    )
    created_b = await client.put(
        f"{API}/projects/{project_b}/composition",
        json=_document([video_b]),
        headers={"If-None-Match": "*"},
    )
    assert created_a.status_code == created_b.status_code == 201
    guessed = await client.get(
        f"{API}/projects/{project_b}/composition/revisions/"
        f"{created_a.json()['revision_id']}"
    )
    assert guessed.status_code == 404


def test_openapi_exposes_typed_composition_contract() -> None:
    response = app.openapi()
    assert "/api/v1/projects/{project_id}/composition" in response["paths"]
    assert (
        "/api/v1/projects/{project_id}/composition/revisions/{revision_id}"
        in response["paths"]
    )
    audio_bed = response["components"]["schemas"]["CompositionAudioBed"]
    mode_ref = audio_bed["properties"]["mode"]["$ref"].rsplit("/", 1)[1]
    assert response["components"]["schemas"][mode_ref]["enum"] == [
        "mix",
        "duck",
        "replace",
    ]


async def test_first_save_race_creates_one_head_and_revision(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project()
    video_id = await _upload(client, project_id, "clip.mp4")
    document = _document([video_id])

    responses = await asyncio.gather(
        *(
            client.put(
                f"{API}/projects/{project_id}/composition",
                json=document,
                headers={"If-None-Match": "*"},
            )
            for _ in range(2)
        )
    )
    assert sorted(response.status_code for response in responses) == [201, 412]
    current = await client.get(f"{API}/projects/{project_id}/composition")
    assert current.status_code == 200
    assert current.json()["revision_number"] == 1


async def test_concurrent_updates_from_same_revision_only_advance_once(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project()
    video_id = await _upload(client, project_id, "clip.mp4")
    first = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=_document([video_id]),
        headers={"If-None-Match": "*"},
    )
    etag = first.headers["etag"]
    docs = [
        _document([video_id]),
        _document([video_id]),
    ]
    docs[0]["canvas"] = {"width": 1280, "height": 720}
    docs[1]["canvas"] = {"width": 1600, "height": 900}
    responses = await asyncio.gather(
        *(
            client.put(
                f"{API}/projects/{project_id}/composition",
                json=document,
                headers={"If-Match": etag},
            )
            for document in docs
        )
    )
    assert sorted(response.status_code for response in responses) == [200, 412]
    current = await client.get(f"{API}/projects/{project_id}/composition")
    assert current.json()["revision_number"] == 2
