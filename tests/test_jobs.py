import asyncio
import io
import uuid

import pytest
from httpx import AsyncClient

from app.api.v1.jobs import get_enqueuer
from app.capabilities import registry
from app.capabilities.context import HandlerResult, JobContext
from app.capabilities.handlers import audio_tts as audio_tts_module
from app.core.database import async_session_factory
from app.main import app
from app.providers.base import BaseInferenceProvider, InferenceRequest
from app.providers.openrouter import AUDIO_BYTES_KEY
from app.services.jobs import execute_job

API = "/api/v1"


async def _project_with_asset(
    client: AsyncClient, make_project, name: str, mime: str
) -> tuple[str, str]:
    pid = await make_project("Jobs")
    files = {"file": (name, io.BytesIO(b"bytes"), mime)}
    aid = (
        await client.post(f"{API}/projects/{pid}/assets", files=files)
    ).json()["id"]
    return pid, aid


async def test_image_caption_job_succeeds(client: AsyncClient, make_project) -> None:
    pid, aid = await _project_with_asset(client, make_project, "pic.png", "image/png")
    resp = await client.post(
        f"{API}/jobs",
        json={"capability_id": "image.caption", "project_id": pid, "asset_id": aid},
    )
    assert resp.status_code == 201
    job_id = resp.json()["id"]

    # Eager enqueuer ran the job synchronously -> already terminal.
    resp = await client.get(f"{API}/jobs/{job_id}")
    job = resp.json()
    assert job["status"] == "succeeded"
    assert job["progress"] == 100
    assert "caption" in job["output"]
    assert job["output"]["provider"] == "mock"


async def test_audio_transcribe_job_succeeds(client: AsyncClient, make_project) -> None:
    pid, aid = await _project_with_asset(client, make_project, "clip.mp3", "audio/mpeg")
    resp = await client.post(
        f"{API}/jobs",
        json={"capability_id": "audio.transcribe", "project_id": pid, "asset_id": aid},
    )
    assert resp.status_code == 201
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded"
    assert "text" in job["output"]


async def test_audio_tts_job_succeeds_with_mock_provider(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("TTS")
    resp = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": "audio.tts",
            "project_id": pid,
            "input": {"text": "Hello there", "voice": "alloy"},
        },
    )
    assert resp.status_code == 201
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded"
    assert job["output"]["provider"] == "mock"
    assert job["output"]["text"] == "Hello there"
    # The mock provider doesn't synthesize audio, so no derived asset.
    assert "outputs" not in job["output"]


async def test_audio_tts_job_persists_derived_audio_asset(
    client: AsyncClient, make_project, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_audio = b"\xff\xfb\x90\x00fake-mp3-bytes"

    class _FakeOpenRouterProvider(BaseInferenceProvider):
        name = "openrouter"

        async def infer(self, request: InferenceRequest) -> dict:
            return {
                AUDIO_BYTES_KEY: fake_audio,
                "voice": request.payload.get("voice", "alloy"),
                "provider": self.name,
            }

    monkeypatch.setattr(audio_tts_module, "get_provider", lambda: _FakeOpenRouterProvider())

    pid = await make_project("TTS Real")
    resp = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": "audio.tts",
            "project_id": pid,
            "input": {"text": "Museum highlights"},
        },
    )
    assert resp.status_code == 201
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded", job.get("error")
    assert job["output"]["provider"] == "openrouter"
    assert AUDIO_BYTES_KEY not in job["output"]
    outputs = job["output"]["outputs"]
    assert len(outputs) == 1
    assert outputs[0]["media_type"] == "audio"

    output_asset = (
        await client.get(f"{API}/assets/{outputs[0]['asset_id']}")
    ).json()
    assert output_asset["is_intermediate"] is False
    assert output_asset["temp_expires_at"] is None
    normal_assets = (await client.get(f"{API}/projects/{pid}/assets")).json()
    assert outputs[0]["asset_id"] in {asset["id"] for asset in normal_assets}

    download = await client.get(f"{API}/assets/{outputs[0]['asset_id']}/download")
    assert download.status_code == 200
    assert download.content == fake_audio


async def test_not_implemented_capability_returns_payload(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("NI")
    resp = await client.post(
        f"{API}/jobs",
        json={"capability_id": "image.restore.face", "project_id": pid, "input": {}},
    )
    assert resp.status_code == 201
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded"
    assert job["output"]["status"] == "not_implemented"


async def test_invalid_capability_rejected(client: AsyncClient) -> None:
    resp = await client.post(
        f"{API}/jobs", json={"capability_id": "no.such.capability"}
    )
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "capability_id,job_input",
    [
        ("image.resize", {"width": 64}),
        ("image.resize", {"width": "wide", "height": 64}),
        ("audio.mix", {"music_asset_id": str(uuid.uuid4()), "mode": "overlay"}),
        ("video.concat", {"asset_ids": [str(uuid.uuid4())]}),
        ("video.concat", {"asset_ids": ["not-a-uuid", str(uuid.uuid4())]}),
        ("image.format", {"format": "bmp"}),
        ("video.transcode", {"format": "avi"}),
        ("video.transcode", {"format": "mp4", "video_codec": "h264"}),
        ("video.transcode", {"format": "mp4", "audio_codec": "mp3"}),
        ("video.transcode", {"format": "webm", "video_codec": "libx264"}),
        ("video.transcode", {"format": "webm", "audio_codec": "aac"}),
    ],
)
async def test_direct_job_rejects_invalid_canonical_input(
    client: AsyncClient, make_project, capability_id: str, job_input: dict
) -> None:
    pid = await make_project("schema-validation")
    response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": capability_id,
            "project_id": pid,
            "input": job_input,
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_job_submission_and_api_remain_async_while_jobs_run(
    client: AsyncClient, make_project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model the production queue boundary without relying on timing or Redis."""
    pid, aid = await _project_with_asset(
        client, make_project, "pic.png", "image/png"
    )
    release = asyncio.Event()
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    started_count = 0
    tasks: list[asyncio.Task[None]] = []

    class _BlockingHandler:
        async def run(self, _ctx: JobContext) -> HandlerResult:
            nonlocal started_count
            started_count += 1
            (first_started if started_count == 1 else second_started).set()
            await release.wait()
            return HandlerResult(data={"completed": True})

    monkeypatch.setitem(registry._HANDLERS, "image.caption", _BlockingHandler())

    async def _run_in_worker(job_id: uuid.UUID) -> None:
        async with async_session_factory() as session:
            await execute_job(session, job_id)

    async def _background_enqueue(job_id: uuid.UUID) -> None:
        tasks.append(asyncio.create_task(_run_in_worker(job_id)))

    app.dependency_overrides[get_enqueuer] = lambda: _background_enqueue
    try:
        first = await client.post(
            f"{API}/jobs",
            json={
                "capability_id": "image.caption",
                "project_id": pid,
                "asset_id": aid,
            },
        )
        assert first.status_code == 201
        assert first.json()["status"] == "queued"
        first_id = first.json()["id"]
        await asyncio.wait_for(first_started.wait(), timeout=2)

        # Independent authorized routes and polling remain responsive while the
        # handler is deliberately suspended in RUNNING state.
        assert (await client.get(f"{API}/projects/{pid}")).status_code == 200
        assert (await client.get(f"{API}/projects/{pid}/assets")).status_code == 200
        detail = await client.get(f"{API}/jobs/{first_id}")
        assert detail.status_code == 200
        assert detail.json()["status"] == "running"
        listing = await client.get(f"{API}/projects/{pid}/jobs")
        assert listing.status_code == 200
        assert listing.json()["items"][0]["status"] == "running"

        # Submission is not serialized behind the first execution. The second
        # worker task reaches the same controlled handler before either is released.
        second = await client.post(
            f"{API}/jobs",
            json={
                "capability_id": "image.caption",
                "project_id": pid,
                "asset_id": aid,
            },
        )
        assert second.status_code == 201
        assert second.json()["status"] == "queued"
        second_id = second.json()["id"]
        await asyncio.wait_for(second_started.wait(), timeout=2)

        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
        assert (await client.get(f"{API}/jobs/{first_id}")).json()["status"] == "succeeded"
        assert (await client.get(f"{API}/jobs/{second_id}")).json()["status"] == "succeeded"
    finally:
        release.set()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def test_cancel_terminal_job_rejected(client: AsyncClient, make_project) -> None:
    pid, aid = await _project_with_asset(client, make_project, "pic.png", "image/png")
    resp = await client.post(
        f"{API}/jobs",
        json={"capability_id": "image.caption", "project_id": pid, "asset_id": aid},
    )
    job_id = resp.json()["id"]  # eager enqueuer already ran it to "succeeded"

    cancel = await client.post(f"{API}/jobs/{job_id}/cancel")
    assert cancel.status_code == 422
    assert cancel.json()["error"]["code"] == "validation_error"

    # Unchanged.
    job = (await client.get(f"{API}/jobs/{job_id}")).json()
    assert job["status"] == "succeeded"


async def test_cancel_queued_job(client: AsyncClient, make_project) -> None:
    pid, aid = await _project_with_asset(client, make_project, "pic.png", "image/png")

    # Skip the eager enqueuer for this one request so the job stays "queued"
    # (the fixture's own override wins on the next test regardless).
    async def _never_runs(_job_id: uuid.UUID) -> None:
        return None

    app.dependency_overrides[get_enqueuer] = lambda: _never_runs
    resp = await client.post(
        f"{API}/jobs",
        json={"capability_id": "image.caption", "project_id": pid, "asset_id": aid},
    )
    job_id = resp.json()["id"]
    assert resp.json()["status"] == "queued"

    cancel = await client.post(f"{API}/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "cancelled"
    assert cancel.json()["error"] == "Cancelled by user"

    job = (await client.get(f"{API}/jobs/{job_id}")).json()
    assert job["status"] == "cancelled"


async def test_cancel_unknown_job_404(client: AsyncClient) -> None:
    resp = await client.post(f"{API}/jobs/{uuid.uuid4()}/cancel")
    assert resp.status_code == 404


async def test_list_jobs_for_project(client: AsyncClient, make_project) -> None:
    pid, aid = await _project_with_asset(client, make_project, "pic.png", "image/png")
    await client.post(
        f"{API}/jobs",
        json={"capability_id": "image.caption", "project_id": pid, "asset_id": aid},
    )
    resp = await client.get(f"{API}/projects/{pid}/jobs")
    assert resp.status_code == 200
    page = resp.json()
    assert page["total"] == 1
    assert page["limit"] == 50
    assert page["offset"] == 0
    assert len(page["items"]) == 1


async def test_list_jobs_pagination(client: AsyncClient, make_project) -> None:
    pid, aid = await _project_with_asset(client, make_project, "pic.png", "image/png")
    for _ in range(5):
        await client.post(
            f"{API}/jobs",
            json={"capability_id": "image.caption", "project_id": pid, "asset_id": aid},
        )

    # First page of 2.
    page1 = (await client.get(f"{API}/projects/{pid}/jobs?limit=2&offset=0")).json()
    assert page1["total"] == 5
    assert page1["limit"] == 2
    assert len(page1["items"]) == 2

    # Second page; ids don't overlap the first.
    page2 = (await client.get(f"{API}/projects/{pid}/jobs?limit=2&offset=2")).json()
    assert len(page2["items"]) == 2
    assert {j["id"] for j in page1["items"]}.isdisjoint(j["id"] for j in page2["items"])

    # limit above the cap is rejected.
    assert (await client.get(f"{API}/projects/{pid}/jobs?limit=999")).status_code == 422


async def test_job_history_retains_audit_data_and_filters_by_status(
    client: AsyncClient, make_project
) -> None:
    pid, aid = await _project_with_asset(client, make_project, "pic.png", "image/png")
    other_pid = await make_project("Other project")
    created: list[dict] = []

    for index in range(11):
        response = await client.post(
            f"{API}/jobs",
            json={
                "capability_id": "image.caption",
                "project_id": pid,
                "asset_id": aid,
                "input": {"prompt": f"audit-{index}"},
            },
        )
        assert response.status_code == 201
        created.append(response.json())

    failed_response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": "image.resize",
            "project_id": pid,
            "asset_id": aid,
            "input": {"width": 20, "height": 20},
        },
    )
    assert failed_response.status_code == 201
    failed = (
        await client.get(f"{API}/jobs/{failed_response.json()['id']}")
    ).json()
    assert failed["status"] == "failed"
    created.append(failed)

    history_response = await client.get(f"{API}/projects/{pid}/jobs")
    assert history_response.status_code == 200
    history = history_response.json()
    assert history["total"] == 12
    assert len(history["items"]) == 12
    assert {job["id"] for job in history["items"]} == {
        job["id"] for job in created
    }
    assert [job["created_at"] for job in history["items"]] == sorted(
        (job["created_at"] for job in history["items"]), reverse=True
    )

    oldest = (await client.get(f"{API}/jobs/{created[0]['id']}")).json()
    assert oldest["capability_id"] == "image.caption"
    assert oldest["asset_id"] == aid
    assert oldest["input"] == {"prompt": "audit-0"}
    assert oldest["output"]["provider"] == "mock"
    assert oldest["started_at"] is not None
    assert oldest["finished_at"] is not None

    failed_detail = (await client.get(f"{API}/jobs/{failed['id']}")).json()
    assert failed_detail["error"] == failed["error"]
    assert "convert" in failed_detail["error"]
    assert failed_detail["started_at"] is not None
    assert failed_detail["finished_at"] is not None

    failed_history = (
        await client.get(f"{API}/projects/{pid}/jobs?status=failed")
    ).json()
    assert failed_history["total"] == 1
    assert [job["id"] for job in failed_history["items"]] == [failed["id"]]
    assert (
        await client.get(f"{API}/projects/{pid}/jobs?status=not-a-status")
    ).status_code == 422

    other_history = (await client.get(f"{API}/projects/{other_pid}/jobs")).json()
    assert other_history["total"] == 0
    assert other_history["items"] == []
