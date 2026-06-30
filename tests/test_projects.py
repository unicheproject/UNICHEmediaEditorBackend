from httpx import AsyncClient

from tests.conftest import DEFAULT_ORG_ID

API = "/api/v1"


async def test_project_crud_and_soft_delete(client: AsyncClient) -> None:
    # Create-up: editor creates the project in the (fake) catalogue.
    resp = await client.post(
        f"{API}/projects",
        json={"name": "Heritage Reel", "slug": "heritage-reel", "org_id": DEFAULT_ORG_ID},
    )
    assert resp.status_code == 201, resp.text
    project = resp.json()
    pid = project["id"]
    assert project["name"] == "Heritage Reel"
    assert project["org_id"] == DEFAULT_ORG_ID
    assert project["deleted_at"] is None

    # Picker (built live from the catalogue) lists it.
    resp = await client.get(f"{API}/projects")
    assert resp.status_code == 200
    assert any(p["id"] == pid for p in resp.json())

    # Get (lazy-JIT/access-checked).
    resp = await client.get(f"{API}/projects/{pid}")
    assert resp.status_code == 200

    # Update description (editor-local) + name (proxied to catalogue).
    resp = await client.patch(f"{API}/projects/{pid}", json={"description": "updated"})
    assert resp.status_code == 200
    assert resp.json()["description"] == "updated"

    # Soft delete (proxied to catalogue, then local).
    resp = await client.delete(f"{API}/projects/{pid}")
    assert resp.status_code == 204

    # Now gone from the catalogue -> 404 and absent from the picker.
    resp = await client.get(f"{API}/projects/{pid}")
    assert resp.status_code == 404
    resp = await client.get(f"{API}/projects")
    assert all(p["id"] != pid for p in resp.json())


async def test_get_missing_project_404(client: AsyncClient) -> None:
    resp = await client.get(f"{API}/projects/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


async def test_create_requires_slug_and_org(client: AsyncClient) -> None:
    resp = await client.post(f"{API}/projects", json={"name": "No slug"})
    assert resp.status_code == 422
