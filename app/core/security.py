"""Keycloak JWT validation (OAuth2 resource server) and request principal.

This mirrors the catalogue's resource-server behaviour: validate signature,
issuer, expiry, and the shared platform audience. No client is needed — the
backend only *validates* incoming user tokens; every outbound catalogue call
forwards the same token (see ``app.services.catalogue_client``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jwt
from fastapi import Request
from jwt import PyJWKClient, PyJWKClientError

from app.core.config import settings
from app.core.errors import UnauthorizedError

SERVICE_ACCOUNT_PREFIX = "service-account-"


@dataclass(frozen=True)
class Principal:
    """The authenticated caller, plus the raw token for catalogue calls."""

    subject: str
    token: str
    email: str | None = None
    preferred_username: str | None = None

    @property
    def is_service_account(self) -> bool:
        return bool(
            self.preferred_username
            and self.preferred_username.startswith(SERVICE_ACCOUNT_PREFIX)
        )


_jwks_client: PyJWKClient | None = None
_jwks_issuer: str | None = None


def _issuer() -> str:
    issuer = settings.idp_issuer_uri.rstrip("/")
    if not issuer:
        raise UnauthorizedError("IDP issuer is not configured")
    return issuer


def _jwks() -> PyJWKClient:
    """Lazily build and cache a JWKS client for the configured issuer."""
    global _jwks_client, _jwks_issuer
    issuer = _issuer()
    if _jwks_client is None or _jwks_issuer != issuer:
        _jwks_client = PyJWKClient(
            f"{issuer}/protocol/openid-connect/certs",
            cache_keys=True,
            lifespan=settings.auth_jwks_cache_seconds,
        )
        _jwks_issuer = issuer
    return _jwks_client


def _extract_bearer(request: Request) -> str:
    header = request.headers.get("authorization")
    if not header or not header.lower().startswith("bearer "):
        raise UnauthorizedError("Missing or malformed Authorization header")
    token = header[len("bearer ") :].strip()
    if not token:
        raise UnauthorizedError("Missing bearer token")
    return token


def decode_token(token: str) -> dict:
    """Validate a Keycloak access token; raise UnauthorizedError on any failure."""
    try:
        signing_key = _jwks().get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.required_audience,
            issuer=_issuer(),
            leeway=settings.auth_leeway_seconds,
            options={"require": ["exp", "iss"]},
        )
    except (jwt.InvalidTokenError, PyJWKClientError) as exc:
        raise UnauthorizedError(f"Invalid token: {exc}") from exc


async def get_current_principal(request: Request) -> Principal:
    """FastAPI dependency: validate the bearer token and return the Principal."""
    token = _extract_bearer(request)
    claims = decode_token(token)
    subject = claims.get("sub")
    if not subject:
        raise UnauthorizedError("Token is missing the 'sub' claim")
    return Principal(
        subject=subject,
        token=token,
        email=claims.get("email"),
        preferred_username=claims.get("preferred_username"),
    )
