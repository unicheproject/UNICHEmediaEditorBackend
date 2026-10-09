"""Project-scoped Asset library queries without changing legacy list responses."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.core.database import async_session_factory
from app.core.errors import ForbiddenError, NotFoundError
from app.core.security import get_current_principal
from app.main import app
from app.models.asset import Asset
from app.models.enums import MediaType
from app.models.project import Project
from app.services.catalogue_client import get_catalogue_client

API = "/api/v1"
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


async def _seed(project_id: str, *, count: int = 5) -> list[uuid.UUID]:
    """Fixed order, mixed media, same-name ties and distinct sizes/timestamps."""
    names = ["Bravo.png", "alpha.mp3", "ALPHA.mp4", "delta.srt", "echo.bin"]
    types = list(MediaType)
    ids = [uuid.UUID(int=index + 1) for index in range(count)]
    async with async_session_factory() as session:
        session.add_all([
            Asset(
                id=asset_id, project_id=uuid.UUID(project_id),
                filename=f"stored-{index}.bin",
                original_filename=names[index] if index < len(names) else f"asset-{index}.png",
                media_type=types[index % len(types)], mime_type="application/octet-stream",
                extension="bin", size_bytes=(count - index) * 10,
                storage_path=f"test/{asset_id}", checksum_sha256="0" * 64,
                provenance_kind="original", ai_provenance="none",
                created_at=EPOCH + timedelta(seconds=index),
            )
            for index, asset_id in enumerate(ids)
        ])
        await session.commit()
    return ids


async def _search(client: AsyncClient, project_id: str, **params) -> dict:
    response = await client.get(f"{API}/projects/{project_id}/assets/search", params=params)
    assert response.status_code == 200, response.text
    return response.json()


async def test_empty_search_default_page(client: AsyncClient, make_project) -> None:
    project_id = await make_project()
    assert await _search(client, project_id) == {
        "items": [], "total": 0, "limit": 50, "offset": 0,
    }


async def test_default_page_is_bounded_and_legacy_array_is_unchanged(
    client: AsyncClient, make_project,
) -> None:
    project_id = await make_project()
    ids = await _seed(project_id, count=65)
    first = await _search(client, project_id)
    assert first["total"] == 65 and first["limit"] == 50 and first["offset"] == 0
    assert [item["id"] for item in first["items"]] == [str(aid) for aid in ids[::-1][:50]]
    second = await _search(client, project_id, offset=50)
    assert second["total"] == 65 and second["offset"] == 50
    assert [item["id"] for item in second["items"]] == [str(aid) for aid in ids[::-1][50:]]
    legacy = await client.get(f"{API}/projects/{project_id}/assets")
    assert legacy.status_code == 200
    assert isinstance(legacy.json(), list) and len(legacy.json()) == 65
    assert [item["id"] for item in legacy.json()] == [str(aid) for aid in ids[::-1]]
    assert set(first["items"][0]) == set(legacy.json()[0])


@pytest.mark.parametrize("q", ["alpha", "ALPHA", "  AlPhA  "])
async def test_name_search_is_trimmed_case_insensitive_substring(
    client: AsyncClient, make_project, q: str,
) -> None:
    project_id = await make_project()
    ids = await _seed(project_id)
    page = await _search(client, project_id, q=q)
    assert page["total"] == 2
    assert [item["id"] for item in page["items"]] == [str(ids[2]), str(ids[1])]


@pytest.mark.parametrize("q", ["", "   "])
async def test_empty_query_means_no_name_filter(
    client: AsyncClient, make_project, q: str,
) -> None:
    project_id = await make_project()
    await _seed(project_id)
    assert (await _search(client, project_id, q=q))["total"] == 5


@pytest.mark.parametrize("name, q", [
    ("100%_ready.png", "%"),
    ("100%_ready.png", "_"),
    ("folder/name.png", "/"),
    ("path\\name.png", "\\"),
    ("archive's photo.png", "'"),
    ("Αρχείο.png", "Αρχείο"),
    ("🔎.png", "🔎"),
])
async def test_search_special_characters_are_literal(
    client: AsyncClient, make_project, name: str, q: str,
) -> None:
    project_id = await make_project()
    ids = await _seed(project_id)
    async with async_session_factory() as session:
        (await session.get(Asset, ids[0])).original_filename = name
        await session.commit()
    page = await _search(client, project_id, q=q)
    assert page["total"] == 1
    assert [item["id"] for item in page["items"]] == [str(ids[0])]


async def test_search_uses_renamed_display_name_not_storage_filename(
    client: AsyncClient, make_project,
) -> None:
    project_id = await make_project()
    uploaded = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("old.png", b"test-media", "image/png")},
    )
    assert uploaded.status_code == 201
    asset_id = uploaded.json()["id"]
    renamed = await client.patch(
        f"{API}/assets/{asset_id}", json={"original_filename": "new-title.png"},
    )
    assert renamed.status_code == 200
    assert (await _search(client, project_id, q="old"))["total"] == 0
    page = await _search(client, project_id, q="new-title")
    assert page["total"] == 1 and page["items"][0] == renamed.json()


@pytest.mark.parametrize("media_type", list(MediaType))
async def test_media_filter_is_applied_before_count_and_pagination(
    client: AsyncClient, make_project, media_type: MediaType,
) -> None:
    project_id = await make_project()
    await _seed(project_id)
    first = await _search(client, project_id, media_type=media_type.value, limit=1)
    assert first["total"] == 1
    assert first["items"][0]["media_type"] == media_type.value
    beyond = await _search(client, project_id, media_type=media_type.value, offset=1)
    assert beyond["total"] == 1 and beyond["items"] == []


async def test_combined_filters_and_pagination(client: AsyncClient, make_project) -> None:
    project_id = await make_project()
    ids = await _seed(project_id)
    page = await _search(client, project_id, q="alpha", media_type="video", limit=1)
    assert page["total"] == 1
    assert [item["id"] for item in page["items"]] == [str(ids[2])]
    assert (await _search(client, project_id, q="alpha", media_type="subtitle"))["total"] == 0


@pytest.mark.parametrize("sort_by, ascending", [
    ("created_at", [0, 1, 2, 3, 4]),
    ("name", [1, 2, 0, 3, 4]),
    ("size", [4, 3, 2, 1, 0]),
    ("type", [1, 0, 3, 4, 2]),
])
@pytest.mark.parametrize("sort_order", ["asc", "desc"])
async def test_all_sort_choices(client: AsyncClient, make_project, sort_by, ascending, sort_order):
    project_id = await make_project()
    ids = await _seed(project_id)
    expected = ascending if sort_order == "asc" else ascending[::-1]
    page = await _search(client, project_id, sort_by=sort_by, sort_order=sort_order)
    assert [item["id"] for item in page["items"]] == [str(ids[index]) for index in expected]


@pytest.mark.parametrize("sort_by", ["created_at", "name", "size", "type"])
@pytest.mark.parametrize("sort_order", ["asc", "desc"])
async def test_sort_ties_have_stable_uuid_order_across_pages(
    client: AsyncClient, make_project, sort_by: str, sort_order: str,
) -> None:
    project_id = await make_project()
    ids = await _seed(project_id)
    async with async_session_factory() as session:
        for asset in (await session.scalars(select(Asset))).all():
            asset.created_at = EPOCH
            asset.original_filename = "same.png"
            asset.size_bytes = 1
            asset.media_type = MediaType.image
        await session.commit()
    expected = ids if sort_order == "asc" else ids[::-1]
    actual = []
    for offset in (0, 2, 4):
        page = await _search(
            client, project_id, sort_by=sort_by, sort_order=sort_order, limit=2, offset=offset,
        )
        assert page["total"] == 5
        actual.extend(item["id"] for item in page["items"])
    assert actual == [str(aid) for aid in expected]


async def test_foreign_deleted_and_intermediate_assets_are_filtered_consistently(
    client: AsyncClient, make_project,
) -> None:
    project_id = await make_project()
    other_project_id = await make_project()
    ids = await _seed(project_id)
    async with async_session_factory() as session:
        (await session.get(Asset, ids[0])).deleted_at = EPOCH
        (await session.get(Asset, ids[1])).is_intermediate = True
        (await session.get(Asset, ids[2])).project_id = uuid.UUID(other_project_id)
        (await session.get(Asset, ids[3])).is_intermediate = True
        (await session.get(Asset, ids[3])).deleted_at = EPOCH
        await session.commit()
    page = await _search(client, project_id)
    assert page["total"] == 1
    assert [item["id"] for item in page["items"]] == [str(ids[4])]
    expanded = await _search(client, project_id, include_intermediate="true")
    assert expanded["total"] == 2
    assert [item["id"] for item in expanded["items"]] == [str(ids[4]), str(ids[1])]
    assert (await _search(client, project_id, q="alpha"))["total"] == 0


@pytest.mark.parametrize("params", [
    {"limit": 0}, {"limit": -1}, {"limit": 201}, {"limit": "x"},
    {"offset": -1}, {"offset": "x"}, {"offset": 2147483648},
    {"media_type": "all"}, {"media_type": "invalid"},
    {"sort_by": "storage_path"}, {"sort_by": "name; DROP TABLE assets"},
    {"sort_order": "descending"}, {"include_intermediate": "invalid"},
    {"q": "x" * 513}, {"q": "\x00"}, {"unexpected": "value"},
])
async def test_invalid_query_is_rejected_without_asset_changes(
    client: AsyncClient, make_project, params: dict,
) -> None:
    project_id = await make_project()
    ids = await _seed(project_id)
    response = await client.get(f"{API}/projects/{project_id}/assets/search", params=params)
    assert response.status_code == 422, response.text
    async with async_session_factory() as session:
        assert list((await session.scalars(select(Asset.id).order_by(Asset.id))).all()) == ids
        assert (await session.get(Project, uuid.UUID(project_id))).deleted_at is None


async def test_query_limits_and_offset_beyond_total(client: AsyncClient, make_project) -> None:
    project_id = await make_project()
    await _seed(project_id)
    page = await _search(client, project_id, limit=200, offset=2147483647)
    assert page == {"items": [], "total": 5, "limit": 200, "offset": 2147483647}
    assert (await _search(client, project_id, q="x" * 512))["total"] == 0


@pytest.mark.parametrize("denial", ["unauthenticated", "concealed", "forbidden"])
async def test_search_authorization_precedes_asset_query(
    client: AsyncClient, make_project, monkeypatch: pytest.MonkeyPatch, denial: str,
) -> None:
    project_id = await make_project()
    await _seed(project_id)
    catalogue = app.dependency_overrides[get_catalogue_client]()
    error = ForbiddenError("Project access denied") if denial == "forbidden" else NotFoundError(
        "Project not found",
    )
    lookup = AsyncMock(side_effect=error)
    monkeypatch.setattr(catalogue, "get_project", lookup)
    if denial == "unauthenticated":
        app.dependency_overrides.pop(get_current_principal)
    from app.services import assets as svc

    search = AsyncMock()
    monkeypatch.setattr(svc, "search_assets", search)
    response = await client.get(f"{API}/projects/{project_id}/assets/search", params={"q": "%"})
    assert response.status_code == {
        "unauthenticated": 401, "concealed": 404, "forbidden": 403,
    }[denial]
    search.assert_not_called()
    if denial == "unauthenticated":
        lookup.assert_not_called()
    else:
        lookup.assert_awaited_once()


async def test_missing_project_is_concealed(client: AsyncClient) -> None:
    response = await client.get(f"{API}/projects/{uuid.uuid4()}/assets/search")
    assert response.status_code == 404


def test_asset_search_openapi_contract() -> None:
    document = app.openapi()
    operation = document["paths"][f"{API}/projects/{{project_id}}/assets/search"]["get"]
    parameters = {param["name"]: param["schema"] for param in operation["parameters"]}
    assert parameters["limit"]["default"] == 50
    assert parameters["limit"]["minimum"] == 1 and parameters["limit"]["maximum"] == 200
    assert parameters["offset"]["minimum"] == 0
    assert parameters["offset"]["maximum"] == 2147483647
    assert parameters["sort_by"]["enum"] == ["created_at", "name", "size", "type"]
    assert parameters["sort_order"]["enum"] == ["asc", "desc"]
    assert parameters["include_intermediate"]["default"] is False
    assert "200" in operation["responses"]
    legacy = document["paths"][f"{API}/projects/{{project_id}}/assets"]["get"]
    assert legacy["responses"]["200"]["content"]["application/json"]["schema"]["type"] == "array"
