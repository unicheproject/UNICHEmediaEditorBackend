import io

from httpx import AsyncClient

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


async def test_not_implemented_capability_returns_payload(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("NI")
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
