"""Asset parameter item schemas and execution share one media contract."""

import copy
import io
import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from jsonschema import Draft202012Validator
from sqlalchemy import func, select

from app.agent.catalog import build_catalog, catalog_text
from app.agent.planner import PlanValidationError, validate_plan
from app.agent.schemas import Plan
from app.api.v1.jobs import get_enqueuer
from app.capabilities import registry
from app.capabilities.definitions import CAPABILITIES
from app.capabilities.validation import iter_media_asset_references
from app.core.database import async_session_factory
from app.main import app
from app.models.enums import MediaType
from app.models.job import Job

API = "/api/v1"


def test_registry_asset_parameter_item_contract_matches_runtime() -> None:
    """Introspect declarations, including unavailable operations, without an ID map."""
    for capability in CAPABILITIES:
        Draft202012Validator.check_schema(capability.input_schema)
        if capability.applicability.primary_asset.value == "forbidden":
            assert capability.supported_media_types == []
        for name, schema in capability.input_schema.get("properties", {}).items():
            is_array = schema.get("type") == "array"
            reference_schema = schema.get("items", {}) if is_array else schema
            if name == "asset_ids" or name.endswith("_asset_id"):
                assert "x-media-asset" in reference_schema, (capability.id, name)
            if is_array and reference_schema.get("format") == "uuid":
                assert "x-media-asset" in reference_schema, (capability.id, name)
            if "x-media-asset" not in schema and "x-media-asset" not in reference_schema:
                continue
            semantics = reference_schema["x-media-asset"]
            assert reference_schema["type"] == "string"
            assert reference_schema["format"] == "uuid"
            assert semantics["role"]
            assert semantics["media_types"]
            assert set(semantics["media_types"]) <= {media.value for media in MediaType}
            if is_array:
                assert schema["x-media-asset"] == semantics
            first, second = str(uuid.uuid4()), str(uuid.uuid4())
            values = [first, second, first] if is_array else [first]
            value = values if is_array else first
            references = list(iter_media_asset_references(capability, {name: value}))
            assert [ref.asset_id for ref in references] == values
            assert all(ref.parameter == name for ref in references)
            assert all(ref.role == semantics["role"] for ref in references)
            assert all(ref.media_types == tuple(semantics["media_types"]) for ref in references)


@pytest.mark.parametrize(
    "capability_id,media,min_items",
    [
        ("video.concat", "video", 2),
        ("audio.concat", "audio", 2),
        ("image.slideshow", "image", 1),
        ("video.compose", "video", 1),
    ],
)
async def test_capability_api_preserves_array_item_media_contract(
    client: AsyncClient, capability_id: str, media: str, min_items: int
) -> None:
    detail = await client.get(f"{API}/capabilities/{capability_id}")
    assert detail.status_code == 200
    capability = detail.json()
    field = capability["input_schema"]["properties"]["asset_ids"]
    assert field["type"] == "array"
    assert field["minItems"] == min_items
    assert field["items"]["type"] == "string"
    assert field["items"]["format"] == "uuid"
    assert field["items"]["x-media-asset"]["media_types"] == [media]
    assert field["x-media-asset"] == field["items"]["x-media-asset"]
    assert field["title"] and field["description"]
    assert "asset_ids" in capability["input_schema"]["required"]
    assert capability["supported_media_types"] == [media]
    assert capability["applicability"]["primary_asset"] == "optional"
    assert capability["availability"]["executable"] is True
    assert "x-ui" not in str(capability)
    listing = (await client.get(f"{API}/capabilities")).json()
    assert next(cap for cap in listing if cap["id"] == capability_id) == capability


@pytest.mark.parametrize(
    "capability_id,parameter,media",
    [
        ("audio.mix", "music_asset_id", "audio"),
        ("video.subtitle.embed", "subtitle_asset_id", "subtitle"),
        ("video.compose", "audio_asset_id", "audio"),
        ("video.compose", "subtitle_asset_id", "subtitle"),
        ("image.inpaint", "asset_id", "image"),
    ],
)
async def test_scalar_asset_metadata_remains_truthful(
    client: AsyncClient, capability_id: str, parameter: str, media: str
) -> None:
    response = await client.get(f"{API}/capabilities/{capability_id}")
    assert response.status_code == 200
    schema = response.json()["input_schema"]["properties"][parameter]
    assert schema["type"] == "string"
    assert schema["format"] == "uuid"
    assert schema["x-media-asset"]["media_types"] == [media]


def test_item_metadata_alone_drives_runtime_and_agent_grounding(monkeypatch) -> None:
    capability = registry.get("video.concat").model_copy(deep=True)
    field = capability.input_schema["properties"]["asset_ids"]
    field.pop("x-media-asset")
    asset_id = str(uuid.uuid4())
    references = list(iter_media_asset_references(capability, {"asset_ids": [asset_id] * 2}))
    assert [reference.media_types for reference in references] == [("video",)] * 2

    monkeypatch.setattr(registry, "list_enabled", lambda: [capability])
    assert build_catalog()[0]["input_schema"] == capability.input_schema
    text = catalog_text()
    assert "asset_role=video_segment" in text
    assert "asset_media=['video']" in text
    assert "min_items=2" in text


def test_legacy_array_container_metadata_still_validates() -> None:
    capability = registry.get("video.concat").model_copy(deep=True)
    field = capability.input_schema["properties"]["asset_ids"]
    field["items"].pop("x-media-asset")
    asset_id = str(uuid.uuid4())
    references = list(iter_media_asset_references(capability, {"asset_ids": [asset_id] * 2}))
    assert [reference.media_types for reference in references] == [("video",)] * 2


def test_agent_catalog_carries_item_constraints_without_changing_applicability() -> None:
    catalog = build_catalog()
    for entry in catalog:
        capability = registry.get(entry["capability_id"])
        assert entry["input_schema"] == capability.input_schema
        assert entry["applicability"] == capability.applicability.model_dump(mode="json")
        for field in entry["input_schema"].get("properties", {}).values():
            if field.get("type") == "array" and "x-media-asset" in field:
                assert field["items"]["x-media-asset"] == field["x-media-asset"]
    assert "image.inpaint" not in {entry["capability_id"] for entry in catalog}


@pytest.mark.parametrize(
    "capability_id,valid_name,valid_mime,wrong_name,wrong_mime",
    [
        ("video.concat", "video.mp4", "video/mp4", "audio.mp3", "audio/mpeg"),
        ("audio.concat", "audio.mp3", "audio/mpeg", "video.mp4", "video/mp4"),
        ("image.slideshow", "image.png", "image/png", "video.mp4", "video/mp4"),
        ("video.compose", "video.mp4", "video/mp4", "audio.mp3", "audio/mpeg"),
    ],
)
async def test_mixed_array_media_rejected_before_persistence_or_queue(
    client: AsyncClient,
    make_project,
    capability_id: str,
    valid_name: str,
    valid_mime: str,
    wrong_name: str,
    wrong_mime: str,
) -> None:
    project_id = await make_project("Array media contract")
    asset_ids = []
    for name, mime in [(valid_name, valid_mime), (wrong_name, wrong_mime)]:
        upload = await client.post(
            f"{API}/projects/{project_id}/assets",
            files={"file": (name, io.BytesIO(b"fixture"), mime)},
        )
        assert upload.status_code == 201
        asset_ids.append(upload.json()["id"])
    enqueuer = AsyncMock()
    app.dependency_overrides[get_enqueuer] = lambda: enqueuer
    body = {
        "project_id": project_id,
        "capability_id": capability_id,
        "input": {"asset_ids": asset_ids},
    }
    invalid = await client.post(f"{API}/jobs", json=body)
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_error"
    assert "requires" in invalid.json()["error"]["message"]
    enqueuer.assert_not_awaited()
    async with async_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Job)) == 0

    plan = Plan.model_validate({"type": "plan", "steps": [{
        "id": "compose", "capability_id": capability_id,
        "assets": asset_ids, "params": {},
    }]})
    accepted_media = registry.get(capability_id).input_schema["properties"][
        "asset_ids"
    ]["items"]["x-media-asset"]["media_types"][0]
    scope = {asset_ids[0]: accepted_media, asset_ids[1]: wrong_mime.split("/")[0]}
    with pytest.raises(PlanValidationError, match="needs.*media"):
        validate_plan(plan, scope)

    valid_body = copy.deepcopy(body)
    valid_body["input"]["asset_ids"] = [asset_ids[0], asset_ids[0]]
    valid = await client.post(f"{API}/jobs", json=valid_body)
    assert valid.status_code == 201, valid.text
    assert valid.json()["input"]["asset_ids"] == [asset_ids[0], asset_ids[0]]
    enqueuer.assert_awaited_once_with(uuid.UUID(valid.json()["id"]))
