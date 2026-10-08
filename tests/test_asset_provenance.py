"""Authoritative provenance across publication, secondary inputs and uploads."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select

from app.capabilities import registry
from app.capabilities.context import HandlerResult, OutputFile
from app.core.database import async_session_factory
from app.models.asset import Asset
from app.models.enums import JobStatus, MediaType
from app.models.job import Job
from app.services import jobs as jobs_svc
from app.services.provenance import contributing_asset_ids, output_ai_provenance

API = "/api/v1"
SUBTITLE = b"1\n00:00:00,000 --> 00:00:01,000\nEdited transcript\n"


async def upload(client, project_id, name="source.mp4", **data):
    response = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": (name, SUBTITLE, "application/octet-stream")},
        data=data,
    )
    return response


async def set_ai(asset_id, value):
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(asset_id))
        asset.ai_provenance = value
        await session.commit()


class FileHandler:
    async def run(self, ctx):
        path = ctx.out_path("output.srt")
        path.write_bytes(SUBTITLE)
        return HandlerResult(outputs=[OutputFile(path, path.name, MediaType.subtitle)])


async def execute_output(
    project_id, capability_id, monkeypatch, *, asset_id=None, params=None, intermediate=False
):
    monkeypatch.setitem(registry._HANDLERS, capability_id, FileHandler())

    async def ready(_capability):
        return None

    monkeypatch.setattr(jobs_svc, "require_worker_readiness", ready)
    async with async_session_factory() as session:
        job = Job(
            project_id=uuid.UUID(project_id),
            capability_id=capability_id,
            asset_id=uuid.UUID(asset_id) if asset_id else None,
            input=params or {},
            status=JobStatus.queued,
            progress=0,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job = await jobs_svc.execute_job(session, job.id, is_intermediate_output=intermediate)
        assert job.status == JobStatus.succeeded, job.error
        descriptor = job.output["outputs"][0]
        assert set(descriptor) == {
            "asset_id",
            "filename",
            "media_type",
            "size_bytes",
            "download_path",
        }
        result = await session.get(Asset, uuid.UUID(descriptor["asset_id"]))
        assert result.provenance_kind == "derived"
        assert result.source_job_id == job.id
        assert result.is_intermediate == intermediate
        return result


async def test_upload_list_detail_and_manual_subtitle(client, make_project):
    project_id = await make_project()
    for name in ["photo.png", "manual.srt", "manual.vtt"]:
        response = await upload(client, project_id, name)
        assert response.status_code == 201, response.text
        asset = response.json()
        assert (asset["provenance_kind"], asset["ai_provenance"], asset["source_job_id"]) == (
            "original",
            "none",
            None,
        )
        assert asset["source_asset_id"] is None
        assert (await client.get(f"{API}/assets/{asset['id']}")).json() == asset
    items = (await client.get(f"{API}/projects/{project_id}/assets")).json()
    assert len(items) == 3 and all(a["provenance_kind"] == "original" for a in items)


@pytest.mark.parametrize(
    "state,expected",
    [
        ("none", "none"),
        ("unknown", "unknown"),
        ("generated", "inherited"),
        ("modified", "inherited"),
        ("inherited", "inherited"),
    ],
)
async def test_deterministic_transcode_inheritance(
    client, make_project, monkeypatch, state, expected
):
    project_id = await make_project()
    source = (await upload(client, project_id)).json()
    await set_ai(source["id"], state)
    result = await execute_output(
        project_id, "video.transcode", monkeypatch, asset_id=source["id"], params={"format": "mp4"}
    )
    assert result.ai_provenance == expected
    assert result.source_asset_id == uuid.UUID(source["id"])


@pytest.mark.parametrize(
    "capability_id,params,primary_name,secondary_name",
    [
        ("video.concat", {}, "primary.mp4", "secondary.mp4"),
        ("audio.mix", {"mode": "replace"}, "primary.mp4", "secondary.wav"),
        ("video.compose", {"width": 96, "height": 64}, "primary.mp4", "secondary.wav"),
        ("video.subtitle.embed", {}, "primary.mp4", "secondary.srt"),
        ("image.slideshow", {}, "primary.png", "secondary.png"),
    ],
)
async def test_multi_input_inherits_secondary_ai(
    client,
    make_project,
    monkeypatch,
    capability_id,
    params,
    primary_name,
    secondary_name,
):
    project_id = await make_project()
    primary = (await upload(client, project_id, primary_name)).json()["id"]
    secondary = (await upload(client, project_id, secondary_name)).json()["id"]
    await set_ai(secondary, "generated")
    if capability_id in {"video.concat", "image.slideshow"}:
        params["asset_ids"] = [primary, secondary]
    elif capability_id == "audio.mix":
        params["music_asset_id"] = secondary
    elif capability_id == "video.compose":
        params.update(asset_ids=[primary], audio_asset_id=secondary)
    else:
        params["subtitle_asset_id"] = secondary
    result = await execute_output(
        project_id, capability_id, monkeypatch, asset_id=primary, params=params
    )
    assert result.ai_provenance == "inherited"


@pytest.mark.parametrize(
    "capability_id,name,params,expected",
    [
        ("media.titlecard", None, {"text": "Title"}, "none"),
        ("audio.tts", None, {"text": "Speech"}, "generated"),
        ("image.upscale", "image.png", {}, "modified"),
        ("audio.denoise", "audio.wav", {}, "modified"),
    ],
)
async def test_direct_effect_and_sourceless_outputs(
    client,
    make_project,
    monkeypatch,
    capability_id,
    name,
    params,
    expected,
):
    project_id = await make_project()
    source = (await upload(client, project_id, name)).json()["id"] if name else None
    if source:
        await set_ai(source, "unknown")
    result = await execute_output(
        project_id, capability_id, monkeypatch, asset_id=source, params=params, intermediate=True
    )
    assert result.ai_provenance == expected
    assert result.source_asset_id == (uuid.UUID(source) if source else None)


async def seed_job(
    project_id, *, status=JobStatus.succeeded, capability_id="audio.transcribe", asset_id=None
):
    async with async_session_factory() as session:
        job = Job(
            project_id=uuid.UUID(project_id) if project_id else None,
            asset_id=uuid.UUID(asset_id) if asset_id else None,
            capability_id=capability_id,
            status=status,
            input={},
            output={"text": "Transcript"},
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return str(job.id)


@pytest.mark.parametrize("status", ["queued", "running", "failed", "cancelled"])
async def test_non_successful_materialization_rejected(client, make_project, status):
    project_id = await make_project()
    job_id = await seed_job(project_id, status=JobStatus(status))
    response = await upload(client, project_id, "result.srt", derived_from_job_id=job_id)
    assert response.status_code == 422, response.text
    assert (await client.get(f"{API}/projects/{project_id}/assets")).json() == []


@pytest.mark.parametrize(
    "case", ["forged", "cross_project", "unscoped", "ineligible", "wrong_media"]
)
async def test_invalid_materialization_rejected(client, make_project, case):
    project_id = await make_project()
    owner = await make_project() if case == "cross_project" else project_id
    job_id = (
        str(uuid.uuid4())
        if case == "forged"
        else await seed_job(
            None if case == "unscoped" else owner,
            capability_id="image.caption" if case == "ineligible" else "audio.transcribe",
        )
    )
    response = await upload(
        client,
        project_id,
        "result.png" if case == "wrong_media" else "result.srt",
        derived_from_job_id=job_id,
    )
    assert response.status_code == (404 if case in {"forged", "cross_project", "unscoped"} else 422)
    async with async_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Asset)) == 0


async def test_denied_project_materialization_is_concealed(client, make_project, monkeypatch):
    from app.services.projects import require_project_access

    project_id = await make_project()
    job_id = await seed_job(project_id)

    async def denied(*args, **kwargs):
        from app.core.errors import NotFoundError

        raise NotFoundError("Project not found")

    monkeypatch.setattr("app.services.projects.require_project_access", denied)
    response = await upload(client, project_id, "result.srt", derived_from_job_id=job_id)
    assert response.status_code == 404
    monkeypatch.setattr("app.services.projects.require_project_access", require_project_access)
    assert (await client.get(f"{API}/projects/{project_id}/assets")).json() == []


@pytest.mark.parametrize("output", [None, {}, {"text": None}, {"text": 5}])
async def test_legacy_success_without_valid_transcription_result_is_rejected(
    client, make_project, output,
):
    project_id = await make_project()
    job_id = await seed_job(project_id)
    async with async_session_factory() as session:
        job = await session.get(Job, uuid.UUID(job_id))
        job.output = output
        await session.commit()
    response = await upload(client, project_id, "result.srt", derived_from_job_id=job_id)
    assert response.status_code == 422, response.text
    assert (await client.get(f"{API}/projects/{project_id}/assets")).json() == []


async def test_empty_transcription_result_remains_materializable(client, make_project):
    project_id = await make_project()
    job_id = await seed_job(project_id)
    async with async_session_factory() as session:
        job = await session.get(Job, uuid.UUID(job_id))
        job.output = {"text": ""}
        await session.commit()
    response = await upload(client, project_id, "silence.srt", derived_from_job_id=job_id)
    assert response.status_code == 201, response.text
    assert response.json()["ai_provenance"] == "generated"


@pytest.mark.parametrize("name", ["edited.srt", "edited.vtt"])
async def test_transcription_materialization(client, make_project, monkeypatch, name):
    from app.capabilities.handlers import base as handler_base
    from app.core.config import settings
    from app.providers.base import BaseInferenceProvider

    class FakeProvider(BaseInferenceProvider):
        name = "http"

        async def infer(self, request):
            return {"text": "Transcript"}

    monkeypatch.setattr(settings, "inference_provider", "http")
    monkeypatch.setattr(settings, "inference_base_url", "http://inference.test")
    monkeypatch.setattr(handler_base, "get_provider", lambda: FakeProvider())
    project_id = await make_project()
    source_id = (await upload(client, project_id, "speech.wav")).json()["id"]
    response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": "audio.transcribe",
            "asset_id": source_id,
            "input": {},
        },
    )
    assert response.status_code == 201, response.text
    job = response.json()
    assert job["status"] == "succeeded" and isinstance(job["output"]["text"], str)
    response = await upload(client, project_id, name, derived_from_job_id=job["id"])
    assert response.status_code == 201, response.text
    result = response.json()
    assert (result["provenance_kind"], result["ai_provenance"]) == ("derived", "generated")
    assert result["source_job_id"] == job["id"] and result["source_asset_id"] == source_id


@pytest.mark.parametrize(
    "field",
    [
        "provenance_kind",
        "ai_provenance",
        "ai_generated",
        "source_job_id",
        "source_asset_id",
    ],
)
async def test_client_cannot_set_provenance(client, make_project, field):
    project_id = await make_project()
    response = await upload(client, project_id, "photo.png", **{field: "forged"})
    assert response.status_code == 422
    assert (await client.get(f"{API}/projects/{project_id}/assets")).json() == []


async def test_registry_public_semantics(client):
    expected = {
        "audio.transcribe": "generated",
        "subtitle.autogenerate": "generated",
        "audio.tts": "generated",
        "audio.music.generate": "generated",
        "image.caption": "generated",
        "image.tag": "generated",
        "image.upscale": "modified",
        "audio.denoise": "modified",
        "audio.separate.stems": "modified",
        "image.background.remove": "modified",
        "image.inpaint": "modified",
        "image.restore.face": "modified",
        "video.upscale": "modified",
    }
    capabilities = (await client.get(f"{API}/capabilities")).json()
    assert {
        cap["id"]: cap["ai_content_effect"]
        for cap in capabilities
        if cap["ai_content_effect"] != "none"
    } == expected
    assert {
        cap["id"]: cap["client_materialization_media_types"]
        for cap in capabilities
        if cap["client_materialization_media_types"]
    } == {"audio.transcribe": ["subtitle"]}


async def test_reference_reader_ignores_unrelated_ids_and_ai_precedence(client, make_project):
    project_id = await make_project()
    primary = (await upload(client, project_id)).json()["id"]
    unknown = (await upload(client, project_id)).json()["id"]
    ai = (await upload(client, project_id)).json()["id"]
    await set_ai(unknown, "unknown")
    await set_ai(ai, "inherited")
    job = Job(
        asset_id=uuid.UUID(primary),
        capability_id="video.concat",
        input={
            "asset_ids": [unknown, ai],
            "unrelated_asset_id": str(uuid.uuid4()),
        },
    )
    ids = contributing_asset_ids(job)
    assert set(ids) == {uuid.UUID(i) for i in [primary, unknown, ai]}
    async with async_session_factory() as session:
        assert (
            await output_ai_provenance(
                session,
                project_id=uuid.UUID(project_id),
                capability_ids=["video.concat"],
                asset_ids=ids,
            )
            == "inherited"
        )


@pytest.mark.parametrize("ai_state,expected", [("none", "none"), ("generated", "inherited")])
async def test_agent_fused_group_uses_logical_operations_and_source(
    client,
    make_project,
    monkeypatch,
    ai_state,
    expected,
):
    from pathlib import Path

    from app.agent.executor import execute_plan
    from tests.test_job_fencing import setup_agent

    async def render(source, destination, *_args):
        destination.write_bytes(Path(source).read_bytes())
        return False

    monkeypatch.setattr("app.agent.render.render_group", render)
    project_id, plan_id, attempt_id = await setup_agent(
        client, make_project, monkeypatch, fused=True
    )
    async with async_session_factory() as session:
        source = await session.scalar(select(Asset).where(Asset.project_id == project_id))
        source.ai_provenance = ai_state
        await session.commit()
        plan = await execute_plan(session, plan_id, attempt_id)
        assert plan.status.value == "succeeded", plan.error
        asset = await session.get(Asset, uuid.UUID(plan.result_asset_ids[0]))
        assert asset.provenance_kind == "derived" and asset.ai_provenance == expected
        assert asset.source_job_id == uuid.UUID(plan.step_runs[0]["job_id"])


async def test_agent_direct_ai_uses_capability_not_plan_initiation(
    client, make_project, monkeypatch
):
    from app.agent.schemas import Plan
    from app.core.config import settings

    project_id = await make_project()
    monkeypatch.setattr(settings, "inference_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "test-only-key")
    monkeypatch.setattr(settings, "agent_deterministic_only", False)
    monkeypatch.setitem(registry._HANDLERS, "audio.tts", FileHandler())
    response = await client.post(f"{API}/agent/sessions", json={"project_id": project_id})
    session_id = response.json()["id"]

    async def propose(*_args, **_kwargs):
        return Plan.model_validate(
            {
                "type": "plan",
                "summary": "Generate speech",
                "steps": [
                    {"id": "speech", "capability_id": "audio.tts", "params": {"text": "Speech"}},
                ],
            }
        ), []

    monkeypatch.setattr("app.services.agent.llm.propose", propose)
    response = await client.post(
        f"{API}/agent/sessions/{session_id}/messages", json={"content": "Speech"}
    )
    assert response.status_code == 200, response.text
    plan_id = response.json()["plan"]["id"]
    response = await client.post(f"{API}/agent/plans/{plan_id}/approve")
    assert response.status_code == 200, response.text
    plan = (await client.get(f"{API}/agent/plans/{plan_id}")).json()
    assert plan["status"] == "succeeded", plan.get("error")
    asset = (await client.get(f"{API}/assets/{plan['result_asset_ids'][0]}")).json()
    assert (asset["provenance_kind"], asset["ai_provenance"]) == ("derived", "generated")


@pytest.mark.parametrize("ai_input,expected", [
    ("video", "inherited"), ("audio", "inherited"), ("current_revision_only", "none"),
])
async def test_export_lineage_uses_immutable_snapshot_sources(
    client, make_project, tmp_path, ai_input, expected,
):
    import io
    import wave

    from tests.test_project_export import (
        _document,
        _make_video,
        _queue_export,
        _save,
        _upload_video,
    )

    project_id = await make_project()
    path = tmp_path / "source.mp4"
    _make_video(path)
    old_video = await _upload_video(client, project_id, path)
    new_video = await _upload_video(client, project_id, path)
    await set_ai(new_video, "generated")
    if ai_input == "video":
        await set_ai(old_video, "modified")
    document = _document([old_video])
    if ai_input == "audio":
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(44100)
            audio.writeframes(b"\x00\x00" * 44100)
        response = await client.post(f"{API}/projects/{project_id}/assets", files={
            "file": ("bed.wav", buffer.getvalue(), "audio/wav"),
        })
        assert response.status_code == 201, response.text
        audio_id = response.json()["id"]
        await set_ai(audio_id, "generated")
        document["audio_bed"] = {"asset_id": audio_id, "mode": "mix", "volume": 0.3}
    revision = await _save(client, project_id, document)
    job = await _queue_export(client, project_id, revision["revision_id"], preset="web")
    current = await client.get(f"{API}/projects/{project_id}/composition")
    await _save(client, project_id, _document([new_video]), etag=current.headers["etag"])
    async with async_session_factory() as session:
        executed = await jobs_svc.execute_job(session, uuid.UUID(job["id"]))
        assert executed.status == JobStatus.succeeded, executed.error
        asset = await session.get(Asset, uuid.UUID(executed.output["outputs"][0]["asset_id"]))
        assert (asset.provenance_kind, asset.ai_provenance, asset.source_job_id) == (
            "derived", expected, executed.id,
        )
