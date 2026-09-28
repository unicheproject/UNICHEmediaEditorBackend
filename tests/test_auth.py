"""Auth boundary tests (no principal override here — the real dependency runs)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import ASGITransport, AsyncClient

from app.core import security
from app.core.config import settings
from app.core.errors import UnauthorizedError
from app.main import app

API = "/api/v1"


@pytest.fixture
def jwt_signing_key(monkeypatch: pytest.MonkeyPatch) -> rsa.RSAPrivateKey:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signing_key = SimpleNamespace(key=private_key.public_key())
    jwks = SimpleNamespace(get_signing_key_from_jwt=lambda _token: signing_key)
    monkeypatch.setattr(security, "_jwks", lambda: jwks)
    monkeypatch.setattr(settings, "idp_issuer_uri", "https://idp.test/realms/uniche")
    monkeypatch.setattr(settings, "required_audience", "uniche-platform")
    monkeypatch.setattr(settings, "auth_leeway_seconds", 0)
    return private_key


def _access_token(
    private_key: rsa.RSAPrivateKey, *, expires_at: datetime
) -> str:
    return jwt.encode(
        {
            "sub": "user-1",
            "iss": "https://idp.test/realms/uniche",
            "aud": "uniche-platform",
            "exp": expires_at,
        },
        private_key,
        algorithm="RS256",
    )


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


async def test_api_accepts_access_token_before_expiry(
    jwt_signing_key: rsa.RSAPrivateKey,
) -> None:
    token = _access_token(
        jwt_signing_key, expires_at=datetime.now(UTC) + timedelta(minutes=1)
    )
    app.dependency_overrides.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            f"{API}/capabilities", headers={"Authorization": f"Bearer {token}"}
        )
    assert resp.status_code == 200


async def test_api_rejects_expired_access_token(
    jwt_signing_key: rsa.RSAPrivateKey,
) -> None:
    token = _access_token(
        jwt_signing_key, expires_at=datetime.now(UTC) - timedelta(seconds=1)
    )
    app.dependency_overrides.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            f"{API}/projects", headers={"Authorization": f"Bearer {token}"}
        )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_decode_token_rejects_expired_access_token(
    jwt_signing_key: rsa.RSAPrivateKey,
) -> None:
    token = _access_token(
        jwt_signing_key, expires_at=datetime.now(UTC) - timedelta(seconds=1)
    )

    with pytest.raises(UnauthorizedError, match="Signature has expired"):
        security.decode_token(token)
