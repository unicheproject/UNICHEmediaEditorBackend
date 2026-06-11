from httpx import AsyncClient

API = "/api/v1"


async def test_project_crud_and_soft_delete(client: AsyncClient) -> None:
    # Create
    resp = await client.post(f"{API}/projects", json={"name": "Heritage Reel"})
    assert resp.status_code == 201
    project = resp.json()
    pid = project["id"]
    assert project["name"] == "Heritage Reel"
    assert project["deleted_at"] is None

    # List
    resp = await client.get(f"{API}/projects")
    assert resp.status_code == 200
    assert any(p["id"] == pid for p in resp.json())

    # Get
    resp = await client.get(f"{API}/projects/{pid}")
    assert resp.status_code == 200

    # Update
    resp = await client.patch(f"{API}/projects/{pid}", json={"description": "updated"})
    assert resp.status_code == 200
    assert resp.json()["description"] == "updated"

    # Soft delete
    resp = await client.delete(f"{API}/projects/{pid}")
    assert resp.status_code == 204

    # Now hidden
    resp = await client.get(f"{API}/projects/{pid}")
    assert resp.status_code == 404
    resp = await client.get(f"{API}/projects")
    assert all(p["id"] != pid for p in resp.json())


async def test_get_missing_project_404(client: AsyncClient) -> None:
    resp = await client.get(f"{API}/projects/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
