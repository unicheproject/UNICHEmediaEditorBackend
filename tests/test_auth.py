"""Auth boundary tests (no principal override here — the real dependency runs)."""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from app.main import app

API = "/api/v1"


async def test_health_is_public() -> None:
    app.dependency_overrides.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200


async def test_api_requires_a_token() -> None:
    app.dependency_overrides.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(f"{API}/projects")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


async def test_api_rejects_malformed_bearer() -> None:
    app.dependency_overrides.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(f"{API}/projects", headers={"Authorization": "Bearer not-a-jwt"})
    assert resp.status_code == 401
