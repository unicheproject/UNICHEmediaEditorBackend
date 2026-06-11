import io

from httpx import AsyncClient

API = "/api/v1"


async def _project_with_asset(client: AsyncClient, name: str, mime: str) -> tuple[str, str]:
    pid = (await client.post(f"{API}/projects", json={"name": "Jobs"})).json()["id"]
    files = {"file": (name, io.BytesIO(b"bytes"), mime)}
    aid = (
        await client.post(f"{API}/projects/{pid}/assets", files=files)
    ).json()["id"]
    return pid, aid


async def test_image_caption_job_succeeds(client: AsyncClient) -> None:
    pid, aid = await _project_with_asset(client, "pic.png", "image/png")
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


async def test_audio_transcribe_job_succeeds(client: AsyncClient) -> None:
    pid, aid = await _project_with_asset(client, "clip.mp3", "audio/mpeg")
    resp = await client.post(
        f"{API}/jobs",
        json={"capability_id": "audio.transcribe", "project_id": pid, "asset_id": aid},
    )
    assert resp.status_code == 201
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded"
    assert "text" in job["output"]


async def test_not_implemented_capability_returns_payload(client: AsyncClient) -> None:
    pid = (await client.post(f"{API}/projects", json={"name": "NI"})).json()["id"]
    resp = await client.post(
        f"{API}/jobs",
        json={"capability_id": "image.upscale", "project_id": pid, "input": {}},
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


async def test_list_jobs_for_project(client: AsyncClient) -> None:
    pid, aid = await _project_with_asset(client, "pic.png", "image/png")
    await client.post(
        f"{API}/jobs",
        json={"capability_id": "image.caption", "project_id": pid, "asset_id": aid},
    )
    resp = await client.get(f"{API}/projects/{pid}/jobs")
    assert resp.status_code == 200
    assert len(resp.json()) == 1
