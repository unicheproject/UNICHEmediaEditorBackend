import io
import uuid

import pytest
from httpx import AsyncClient

from app.agent.executor import execute_plan
from app.api.v1.jobs import get_enqueuer
from app.core.database import async_session_factory
from app.main import app
from app.models.agent import AgentPlan, AgentPlanStatus, AgentSession
from app.models.job import Job, JobStatus
from app.services.catalogue_client import get_catalogue_client
from app.services.jobs import execute_job

API = "/api/v1"


async def _upload(
    client: AsyncClient, project_id: str, name: str, mime: str
) -> str:
    response = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": (name, io.BytesIO(b"asset-bytes"), mime)},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def _never_enqueue(_job_id: uuid.UUID) -> None:
    return None


async def test_job_submission_authorizes_explicit_project_before_asset_resolution(
    client: AsyncClient, make_project
) -> None:
    project_a = await make_project("unauthorized-A")
    project_b = await make_project("foreign-B")
    foreign_asset = await _upload(client, project_b, "foreign.mp4", "video/mp4")

    # The fixture principal initially has access to both projects. Remove A
    # from the Catalogue view to exercise the actual unauthorised API path.
    catalogue = app.dependency_overrides[get_catalogue_client]()
    catalogue.projects.pop(project_a)
    app.dependency_overrides[get_enqueuer] = lambda: _never_enqueue

    async def submit(asset_id: str) -> dict:
        response = await client.post(
            f"{API}/jobs",
            json={
                "capability_id": "video.resize",
                "project_id": project_a,
                "asset_id": asset_id,
                "input": {"width": 320, "height": 240},
            },
        )
        return {"status": response.status_code, **response.json()["error"]}

    missing = await submit(str(uuid.uuid4()))
    foreign = await submit(foreign_asset)

    # Before the fix, these were 404/not_found and 422/validation_error,
    # respectively. Both now fail at the project authorization boundary.
    assert missing["status"] == 404
    assert missing["code"] == "not_found"
    assert foreign == missing


async def test_unauthorized_explicit_project_hides_all_asset_reference_states(
    client: AsyncClient, make_project
) -> None:
    project_a = await make_project("unauthorized-reference-A")
    project_b = await make_project("reference-B")
    own_video = await _upload(client, project_a, "own.mp4", "video/mp4")
    foreign_video = await _upload(client, project_b, "foreign.mp4", "video/mp4")
    foreign_audio = await _upload(client, project_b, "music.mp3", "audio/mpeg")
    foreign_subtitle = await _upload(
        client, project_b, "captions.srt", "application/x-subrip"
    )
    deleted = await _upload(client, project_b, "deleted.mp4", "video/mp4")
    assert (await client.delete(f"{API}/assets/{deleted}")).status_code == 204

    catalogue = app.dependency_overrides[get_catalogue_client]()
    catalogue.projects.pop(project_a)
    app.dependency_overrides[get_enqueuer] = lambda: _never_enqueue

    missing = str(uuid.uuid4())

    async def submit(body: dict) -> dict:
        response = await client.post(f"{API}/jobs", json=body)
        return {"status": response.status_code, **response.json()["error"]}

    variants = [
        # Primary asset_id.
        (
            lambda ref: {
                "capability_id": "video.resize",
                "project_id": project_a,
                "asset_id": ref,
                "input": {"width": 320, "height": 240},
            },
            [foreign_video, missing, deleted],
        ),
        # Ordered input.asset_ids.
        (
            lambda ref: {
                "capability_id": "video.concat",
                "project_id": project_a,
                "input": {"asset_ids": [ref, ref]},
            },
            [foreign_video, missing],
        ),
        # Named secondary asset references.
        (
            lambda ref: {
                "capability_id": "audio.mix",
                "project_id": project_a,
                "asset_id": own_video,
                "input": {"music_asset_id": ref},
            },
            [foreign_audio, missing],
        ),
        (
            lambda ref: {
                "capability_id": "video.compose",
                "project_id": project_a,
                "input": {"asset_ids": [own_video], "audio_asset_id": ref},
            },
            [foreign_audio, missing],
        ),
        (
            lambda ref: {
                "capability_id": "video.compose",
                "project_id": project_a,
                "input": {"asset_ids": [own_video], "subtitle_asset_id": ref},
            },
            [foreign_subtitle, missing],
        ),
    ]

    for make_body, references in variants:
        errors = [await submit(make_body(reference)) for reference in references]
        assert all(error == errors[0] for error in errors)
        assert errors[0]["status"] == 404
        assert errors[0]["code"] == "not_found"


async def test_job_project_is_inferred_and_authorized_when_omitted(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("inferred-project")
    video = await _upload(client, project_id, "video.mp4", "video/mp4")
    app.dependency_overrides[get_enqueuer] = lambda: _never_enqueue

    response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": "video.resize",
            "asset_id": video,
            "input": {"width": 320, "height": 240},
        },
    )

    assert response.status_code == 201, response.text
    assert response.json()["project_id"] == project_id


@pytest.mark.parametrize(
    ("capability_id", "foreign_name", "foreign_mime", "input_factory"),
    [
        (
            "audio.mix",
            "music.mp3",
            "audio/mpeg",
            lambda own, foreign: {"music_asset_id": foreign},
        ),
        (
            "video.compose",
            "bed.mp3",
            "audio/mpeg",
            lambda own, foreign: {"asset_ids": [own], "audio_asset_id": foreign},
        ),
        (
            "video.compose",
            "captions.srt",
            "application/x-subrip",
            lambda own, foreign: {"asset_ids": [own], "subtitle_asset_id": foreign},
        ),
        (
            "video.concat",
            "foreign.mp4",
            "video/mp4",
            lambda own, foreign: {"asset_ids": [own, foreign]},
        ),
    ],
)
async def test_direct_job_rejects_cross_project_asset_references(
    client: AsyncClient,
    make_project,
    capability_id: str,
    foreign_name: str,
    foreign_mime: str,
    input_factory,
) -> None:
    project_a = await make_project("A")
    project_b = await make_project("B")
    own_video = await _upload(client, project_a, "own.mp4", "video/mp4")
    foreign = await _upload(client, project_b, foreign_name, foreign_mime)

    app.dependency_overrides[get_enqueuer] = lambda: _never_enqueue
    response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": capability_id,
            "project_id": project_a,
            "asset_id": own_video if capability_id == "audio.mix" else None,
            "input": input_factory(own_video, foreign),
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_same_project_secondary_asset_is_accepted(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("same-project")
    video = await _upload(client, project_id, "video.mp4", "video/mp4")
    music = await _upload(client, project_id, "music.mp3", "audio/mpeg")

    app.dependency_overrides[get_enqueuer] = lambda: _never_enqueue
    response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": "audio.mix",
            "project_id": project_id,
            "asset_id": video,
            "input": {"music_asset_id": music},
        },
    )

    assert response.status_code == 201, response.text
    assert response.json()["project_id"] == project_id


@pytest.mark.parametrize(
    ("capability_id", "input_factory"),
    [
        ("audio.mix", lambda primary, wrong: {"music_asset_id": wrong}),
        ("video.concat", lambda primary, wrong: {"asset_ids": [primary, wrong]}),
        (
            "video.compose",
            lambda primary, wrong: {"asset_ids": [primary], "audio_asset_id": wrong},
        ),
        (
            "video.subtitle.embed",
            lambda primary, wrong: {"subtitle_asset_id": wrong},
        ),
    ],
)
async def test_same_project_wrong_media_reference_is_rejected(
    client: AsyncClient, make_project, capability_id: str, input_factory
) -> None:
    project_id = await make_project("wrong-media")
    primary = await _upload(client, project_id, "primary.mp4", "video/mp4")
    wrong = await _upload(client, project_id, "wrong.png", "image/png")

    app.dependency_overrides[get_enqueuer] = lambda: _never_enqueue
    response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": capability_id,
            "project_id": project_id,
            "asset_id": primary
            if capability_id in {"audio.mix", "video.subtitle.embed"}
            else None,
            "input": input_factory(primary, wrong),
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "requires" in response.json()["error"]["message"]


async def test_deleted_secondary_asset_is_rejected(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("deleted-secondary")
    video = await _upload(client, project_id, "video.mp4", "video/mp4")
    music = await _upload(client, project_id, "music.mp3", "audio/mpeg")
    assert (await client.delete(f"{API}/assets/{music}")).status_code == 204

    app.dependency_overrides[get_enqueuer] = lambda: _never_enqueue
    response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": "audio.mix",
            "project_id": project_id,
            "asset_id": video,
            "input": {"music_asset_id": music},
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_worker_rejects_preexisting_cross_project_job(
    client: AsyncClient, make_project
) -> None:
    project_a = await make_project("worker-A")
    project_b = await make_project("worker-B")
    video = await _upload(client, project_a, "video.mp4", "video/mp4")
    music = await _upload(client, project_b, "music.mp3", "audio/mpeg")

    async with async_session_factory() as session:
        job = Job(
            project_id=uuid.UUID(project_a),
            asset_id=uuid.UUID(video),
            capability_id="audio.mix",
            status=JobStatus.queued,
            input={"music_asset_id": music},
            progress=0,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

        completed = await execute_job(session, job.id)
        assert completed.status == JobStatus.failed
        assert "not in the job's project" in (completed.error or "")


async def test_worker_rejects_preexisting_wrong_media_reference(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("worker-media")
    video = await _upload(client, project_id, "video.mp4", "video/mp4")
    image = await _upload(client, project_id, "not-music.png", "image/png")

    async with async_session_factory() as session:
        job = Job(
            project_id=uuid.UUID(project_id),
            asset_id=uuid.UUID(video),
            capability_id="audio.mix",
            status=JobStatus.queued,
            input={"music_asset_id": image},
            progress=0,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

        completed = await execute_job(session, job.id)
        assert completed.status == JobStatus.failed
        assert "requires audio media" in (completed.error or "")


async def test_agent_execution_rejects_cross_project_secondary_asset(
    client: AsyncClient, make_project
) -> None:
    project_a = await make_project("agent-A")
    project_b = await make_project("agent-B")
    video = await _upload(client, project_a, "video.mp4", "video/mp4")
    music = await _upload(client, project_b, "music.mp3", "audio/mpeg")

    async with async_session_factory() as session:
        agent_session = AgentSession(
            project_id=uuid.UUID(project_a),
            asset_ids=[video],
            messages=[],
        )
        session.add(agent_session)
        await session.commit()
        await session.refresh(agent_session)
        plan = AgentPlan(
            session_id=agent_session.id,
            summary="cross-project regression",
            status=AgentPlanStatus.proposed,
            steps=[
                {
                    "id": "mix",
                    "capability_id": "audio.mix",
                    "params": {"music_asset_id": music},
                    "asset": video,
                    "assets": [],
                    "depends_on": [],
                }
            ],
            step_runs=[],
            result_asset_ids=[],
        )
        session.add(plan)
        await session.commit()
        await session.refresh(plan)

        from app.services import agent as agent_svc

        approved, attempt, _, _ = await agent_svc.approve_plan(session, plan.id)
        completed = await execute_plan(session, approved.id, attempt.id)
        assert completed.status == AgentPlanStatus.failed
        assert "not in the job's project" in (completed.error or "")
        assert completed.step_runs == []
