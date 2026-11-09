"""MED/028: built-in, versioned composition templates and template apply."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
import subprocess
import uuid
from importlib import resources
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.api.v1.jobs import get_enqueuer
from app.composition import templates
from app.composition.document import canonical_document
from app.core.database import async_session_factory
from app.core.errors import ValidationError
from app.main import app
from app.models.asset import Asset
from app.services.jobs import execute_job

API = "/api/v1"
NEED_FFMPEG = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("convert")), reason="needs ffmpeg + convert"
)

# Published versions are immutable: change a template by adding a new version.
PUBLISHED_SHA256 = {
    "exhibit-intro/v1.json":
        "e9395e8194ac322c80d072a59847e9318ddd726e04a69303342b5e4dd7c6da25",
    "artefact-spotlight/v1.json":
        "a92a3874a6c54ca25e1ca96ee8fdde4397002a39a68604c69e7457f71f3c810c",
    "event-recap/v1.json":
        "96617ca95b5a6df8d1ac7bc01700d163bc0a9b1d1f3b203e6d3401604697913b",
    "educational-explainer/v1.json":
        "01342a1ba5029bcd73ee707b949c1c74dc42931f3240fa2a93eecc7c80052e67",
}

# Minimal required bindings per template: (slot, media kind, multiple?).
REQUIRED = {
    "exhibit-intro": [("hero_image", "image", False)],
    "artefact-spotlight": [("photos", "image", True)],
    "event-recap": [("main_video", "video", False)],
    "educational-explainer": [("slides", "image", True)],
}
FILE_NAMES = {"image": "photo.png", "video": "clip.mp4", "audio": "voice.wav"}


def _raw(filename: str) -> dict:
    return json.loads(resources.files(templates.__name__).joinpath(filename).read_text())


# --- registry and definitions (no binaries) ----------------------------------


def test_all_four_templates_load_with_their_ids_and_versions() -> None:
    loaded = {(t.template_id, t.version) for t in templates.latest_templates()}
    assert loaded == {(name, 1) for name in REQUIRED}
    assert [entry[2] for entry in templates.TEMPLATE_FILES] == list(PUBLISHED_SHA256)


@pytest.mark.parametrize(("filename", "digest"), PUBLISHED_SHA256.items())
def test_published_template_versions_are_unchanged(filename: str, digest: str) -> None:
    data = resources.files(templates.__name__).joinpath(filename).read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest


def _mutations() -> list[tuple[str, object]]:
    def edit(path: list[object], value: object):
        def apply(doc: dict) -> dict:
            target = doc
            for key in path[:-1]:
                target = target[key]  # type: ignore[index]
            if value is _DELETE:
                del target[path[-1]]  # type: ignore[arg-type]
            else:
                target[path[-1]] = value  # type: ignore[index]
            return doc
        return apply

    return [
        ("unknown top-level field", edit(["script"], "rm -rf /")),
        ("unknown slot field", edit(["slots", "photos", "path"], "../x")),
        ("raw ffmpeg in a clip", edit(["tracks", 0, "clips", 1, "ffmpeg"], ["-vf", "x"])),
        ("asset id instead of slot",
         edit(["tracks", 0, "clips", 1, "asset_id"], str(uuid.uuid4()))),
        ("undefined slot", edit(["tracks", 0, "clips", 1, "slot"], "nowhere")),
        ("unused slot", edit(["slots", "spare"], {"media_type": "image", "required": False,
                                                   "label": "Spare"})),
        ("slot used as the wrong media", edit(["slots", "photos", "media_type"], "video")),
        ("too many per slot", edit(["slots", "photos", "max"], 13)),
        ("image too long", edit(["tracks", 0, "clips", 1, "duration"], 601)),
        ("expression as a duration", edit(["tracks", 0, "clips", 1, "duration"], "5*2")),
        ("bad template id", edit(["template_id"], "../etc")),
        ("unsupported kind", edit(["tracks", 0, "clips", 1, "kind"], "overlay")),
        ("no required visual slot", edit(["slots", "photos", "required"], False)),
        ("unknown suggestion capability",
         edit(["slots", "title_card", "suggestion", "capability_id"], "video.compose")),
        ("invalid suggestion parameter",
         edit(["slots", "title_card", "suggestion", "params"], {"duration": -1})),
        ("missing name", edit(["name"], _DELETE)),
        ("third track", lambda doc: {**doc, "tracks": [*doc["tracks"], doc["tracks"][1]]}),
    ]


_DELETE = object()


@pytest.mark.parametrize(("case", "mutate"), _mutations())
def test_invalid_template_definitions_are_rejected(case: str, mutate) -> None:
    doc = mutate(copy.deepcopy(_raw("artefact-spotlight/v1.json")))
    with pytest.raises(templates.TemplateDefinitionError):
        templates.parse_template(json.dumps(doc))


def test_duplicate_json_keys_are_rejected() -> None:
    raw = json.dumps(_raw("exhibit-intro/v1.json"))
    duplicated = raw.replace('"version": 1,', '"version": 1, "version": 2,', 1)
    with pytest.raises(templates.TemplateDefinitionError, match="duplicate"):
        templates.parse_template(duplicated)


@pytest.mark.parametrize("template_id", list(REQUIRED))
def test_minimal_bindings_compile_to_a_nonempty_composition(template_id: str) -> None:
    template = templates.get_template(template_id, 1)
    assert template is not None
    bindings: dict = {}
    for slot, _media, multiple in REQUIRED[template_id]:
        bindings[slot] = [uuid.uuid4()] if multiple else uuid.uuid4()
    document = templates.compile_template(template, bindings)
    assert len(document.video_track.clips) == 1
    assert document.audio_track is None and document.audio_bed is None


def test_compiled_ids_are_deterministic_and_bindings_change_the_hash() -> None:
    template = templates.get_template("artefact-spotlight", 1)
    assert template is not None
    photos = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
    first = canonical_document(templates.compile_template(template, {"photos": photos}))
    again = canonical_document(templates.compile_template(template, {"photos": list(photos)}))
    assert first == again
    clips = first[0]["tracks"][0]["clips"]
    assert [c["asset_id"] for c in clips] == [str(p) for p in photos]   # order kept
    assert len({c["id"] for c in clips}) == 3
    reordered = canonical_document(
        templates.compile_template(template, {"photos": photos[::-1]})
    )
    assert reordered[1] != first[1]
    other = templates.get_template("educational-explainer", 1)
    assert other is not None
    explainer = canonical_document(templates.compile_template(other, {"slides": photos}))
    assert {c["id"] for c in explainer[0]["tracks"][0]["clips"]}.isdisjoint(
        {c["id"] for c in clips}
    )


@pytest.mark.parametrize(("bindings", "message"), [
    ({}, "is required"),
    ({"photos": []}, "at least one"),
    ({"photos": [str(uuid.uuid4())] * 13}, "at most 12"),
    ({"photos": uuid.uuid4()}, "a list of Asset IDs"),
    ({"photos": [uuid.uuid4()], "music": [uuid.uuid4()]}, "a single Asset ID"),
    ({"photos": [uuid.uuid4()], "logo": uuid.uuid4()}, "Unknown template slot"),
])
def test_invalid_bindings_are_rejected(bindings: dict, message: str) -> None:
    template = templates.get_template("artefact-spotlight", 1)
    assert template is not None
    with pytest.raises(ValidationError, match=message):
        templates.compile_template(template, bindings)


def test_all_optional_slots_bound_compile_in_template_order() -> None:
    template = templates.get_template("artefact-spotlight", 1)
    assert template is not None
    ids = {name: uuid.uuid4() for name in ("title", "video", "voice", "music")}
    photos = [uuid.uuid4(), uuid.uuid4()]
    document = templates.compile_template(template, {
        "title_card": ids["title"], "photos": photos, "main_video": ids["video"],
        "narration": ids["voice"], "music": ids["music"],
    })
    assert [c.asset_id for c in document.video_track.clips] == [
        ids["title"], *photos, ids["video"]
    ]
    assert document.audio_track is not None
    assert document.audio_track.clips[0].asset_id == ids["voice"]
    assert document.audio_track.clips[0].timeline_start == 0
    assert document.audio_bed is not None and document.audio_bed.asset_id == ids["music"]


# --- REST API -------------------------------------------------------------------


async def test_list_and_read_templates(client: AsyncClient) -> None:
    listed = (await client.get(f"{API}/composition-templates")).json()
    assert [t["template_id"] for t in listed] == list(REQUIRED)
    spotlight = next(t for t in listed if t["template_id"] == "artefact-spotlight")
    assert spotlight["version"] == 1 and spotlight["suggested_export_preset"] == "web"
    assert spotlight["slots"]["photos"]["max"] == 12 and "tracks" not in spotlight
    latest = (await client.get(f"{API}/composition-templates/artefact-spotlight")).json()
    pinned = (await client.get(
        f"{API}/composition-templates/artefact-spotlight/versions/1"
    )).json()
    assert latest == pinned and latest["tracks"][0]["clips"][1]["slot"] == "photos"


@pytest.mark.parametrize("path", [
    "/composition-templates/unknown",
    "/composition-templates/artefact-spotlight/versions/2",
])
async def test_unknown_templates_and_versions_return_404(client: AsyncClient, path: str) -> None:
    response = await client.get(f"{API}{path}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "template_not_found"


async def test_template_ids_are_never_paths(client: AsyncClient) -> None:
    response = await client.get(f"{API}/composition-templates/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code in {404, 422}


async def _upload(client: AsyncClient, pid: str, name: str, data: bytes = b"x") -> str:
    response = await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": (name, io.BytesIO(data), "application/octet-stream")},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _apply(client: AsyncClient, pid: str, body: dict, headers: dict | None = None):
    return await client.post(f"{API}/projects/{pid}/composition/template", json=body,
                             headers=headers if headers is not None
                             else {"If-None-Match": "*"})


@pytest.mark.parametrize("template_id", list(REQUIRED))
async def test_each_template_applies_with_its_required_slots(
    client: AsyncClient, make_project, template_id: str
) -> None:
    pid = await make_project(f"template-{template_id}")
    bindings: dict = {}
    for slot, media, multiple in REQUIRED[template_id]:
        asset = await _upload(client, pid, FILE_NAMES[media])
        bindings[slot] = [asset] if multiple else asset
    response = await _apply(client, pid,
                            {"template_id": template_id, "version": 1, "bindings": bindings})
    assert response.status_code == 201, response.text
    saved = response.json()
    assert response.headers["etag"] == f'"{saved["revision_id"]}"'
    assert saved["revision_number"] == 1
    current = await client.get(f"{API}/projects/{pid}/composition")
    assert current.json()["document"] == saved["document"]
    assert "origin" not in saved["document"] and "template" not in json.dumps(saved["document"])


async def test_apply_requires_an_explicit_version(client: AsyncClient, make_project) -> None:
    pid = await make_project("template-version")
    image = await _upload(client, pid, "photo.png")
    for body in ({"template_id": "exhibit-intro", "bindings": {"hero_image": image}},
                 {"template_id": "exhibit-intro", "version": "1",
                  "bindings": {"hero_image": image}}):
        assert (await _apply(client, pid, body)).status_code == 422
    unknown = await _apply(client, pid, {"template_id": "exhibit-intro", "version": 9,
                                         "bindings": {"hero_image": image}})
    assert unknown.status_code == 404


async def test_existing_composition_uses_if_match_and_keeps_history(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("template-existing")
    first, second = await _upload(client, pid, "a.png"), await _upload(client, pid, "b.png")
    body = {"template_id": "educational-explainer", "version": 1,
            "bindings": {"slides": [first]}}
    created = await _apply(client, pid, body)
    assert created.status_code == 201
    etag = created.headers["etag"]

    assert (await _apply(client, pid, body, headers={})).status_code == 428
    assert (await _apply(client, pid, body)).status_code == 412   # If-None-Match: *
    stale = '"00000000-0000-0000-0000-000000000000"'
    assert (await _apply(client, pid, body, headers={"If-Match": stale})).status_code == 412

    same = await _apply(client, pid, body, headers={"If-Match": etag})
    assert same.status_code == 200
    assert same.json()["revision_id"] == created.json()["revision_id"]        # no-op
    assert same.json()["document_hash"] == created.json()["document_hash"]

    changed = await _apply(client, pid, {**body, "bindings": {"slides": [first, second]}},
                           headers={"If-Match": etag})
    assert changed.status_code == 200 and changed.json()["revision_number"] == 2
    assert changed.json()["document_hash"] != created.json()["document_hash"]
    old = await client.get(
        f"{API}/projects/{pid}/composition/revisions/{created.json()['revision_id']}"
    )
    assert old.json()["document"] == created.json()["document"]


@pytest.mark.parametrize("case", ["wrong_media", "other_project", "bad_shape", "unknown_slot",
                                  "missing_required", "too_many"])
async def test_failed_apply_changes_nothing(
    client: AsyncClient, make_project, case: str
) -> None:
    pid = await make_project(f"template-fail-{case}")
    photo = await _upload(client, pid, "photo.png")
    created = await _apply(client, pid, {"template_id": "artefact-spotlight", "version": 1,
                                         "bindings": {"photos": [photo]}})
    etag = created.headers["etag"]
    other_pid = await make_project(f"template-other-{case}")
    bindings = {
        "wrong_media": {"photos": [await _upload(client, pid, "clip.mp4")]},
        "other_project": {"photos": [await _upload(client, other_pid, "photo.png")]},
        "bad_shape": {"photos": "not-a-uuid"},
        "unknown_slot": {"photos": [photo], "logo": photo},
        "missing_required": {"music": None},
        "too_many": {"photos": [photo] * 13},
    }[case]
    response = await _apply(client, pid, {"template_id": "artefact-spotlight", "version": 1,
                                          "bindings": bindings}, headers={"If-Match": etag})
    assert response.status_code == 422, response.text
    current = await client.get(f"{API}/projects/{pid}/composition")
    assert current.headers["etag"] == etag
    assert current.json()["revision_number"] == 1


async def test_apply_to_another_users_project_is_concealed(client: AsyncClient) -> None:
    response = await _apply(client, str(uuid.uuid4()), {
        "template_id": "exhibit-intro", "version": 1, "bindings": {"hero_image": str(uuid.uuid4())},
    })
    assert response.status_code == 404


async def test_plain_put_compositions_are_unaffected(client: AsyncClient, make_project) -> None:
    pid = await make_project("template-legacy")
    video = await _upload(client, pid, "clip.mp4")
    document = {"schema_version": 1, "canvas": {"width": 96, "height": 64},
                "tracks": [{"id": str(uuid.uuid4()), "kind": "video",
                            "clips": [{"id": str(uuid.uuid4()), "asset_id": video}]}],
                "audio_bed": None}
    saved = await client.put(f"{API}/projects/{pid}/composition", json=document,
                             headers={"If-None-Match": "*"})
    assert saved.status_code == 201
    assert saved.json()["document"] == document
    legacy = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert saved.json()["document_hash"] == hashlib.sha256(legacy.encode()).hexdigest()


# --- export of a template-generated composition (tiny media) ------------------


@NEED_FFMPEG
async def test_template_composition_exports(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid = await make_project("template-export")
    photos = []
    for colour in ("red", "blue"):
        png = tmp_path / f"{colour}.png"
        subprocess.run(["convert", "-size", "96x64", f"xc:{colour}", str(png)], check=True,
                       capture_output=True, timeout=30)
        photos.append(await _upload(client, pid, png.name, png.read_bytes()))
    applied = await _apply(client, pid, {"template_id": "educational-explainer", "version": 1,
                                         "bindings": {"slides": photos}})
    assert applied.status_code == 201, applied.text

    async def leave_queued(_job_id: uuid.UUID) -> None:
        return None

    app.dependency_overrides[get_enqueuer] = lambda: leave_queued
    job = (await client.post(f"{API}/jobs", json={
        "capability_id": "project.export", "project_id": pid,
        "input": {"composition_revision_id": applied.json()["revision_id"], "preset": "preview"},
    })).json()
    async with async_session_factory() as session:
        executed = await execute_job(session, uuid.UUID(job["id"]))
        assert executed.status.value == "succeeded", executed.error
        assert executed.output is not None
        output = await session.get(Asset, uuid.UUID(executed.output["outputs"][0]["asset_id"]))
        assert output is not None and output.duration_seconds is not None
        assert output.duration_seconds == pytest.approx(10.0, abs=0.2)  # 2 slides x 5 s
