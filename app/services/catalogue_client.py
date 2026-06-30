"""Client for the UNICHE Catalogue — the authorization authority.

Phase 1 makes only *user-token* calls: every method forwards the logged-in
user's bearer token, so the catalogue authorizes naturally and no service
account / confidential client is required. The background reconcile sweep
(which would need a service account) is deferred to Phase 2.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings
from app.core.errors import (
    AppError,
    ForbiddenError,
    NotFoundError,
    ProviderError,
    UnauthorizedError,
    ValidationError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


def _raise_for_status(resp: httpx.Response, *, context: str) -> None:
    """Map catalogue HTTP errors onto our AppError hierarchy."""
    if resp.status_code < 400:
        return
    if resp.status_code == 401:
        raise UnauthorizedError(f"Catalogue rejected the token ({context})")
    if resp.status_code == 403:
        raise ForbiddenError(f"Not permitted by the catalogue ({context})")
    if resp.status_code == 404:
        raise NotFoundError(f"Not found in the catalogue ({context})")
    if resp.status_code in (400, 409, 422):
        # Surface the catalogue's validation message where possible.
        message = f"Catalogue rejected the request ({context})"
        try:
            body = resp.json()
            detail = body.get("message") or body.get("error") or body.get("detail")
            if detail:
                message = f"{message}: {detail}"
        except ValueError:
            pass
        raise ValidationError(message)
    raise ProviderError(
        f"Catalogue returned {resp.status_code} ({context}): {resp.text[:300]}"
    )


class CatalogueClient:
    """Thin async wrapper over the catalogue REST API (httpx)."""

    def __init__(self, base_url: str, timeout: float) -> None:
        self._base = f"{base_url.rstrip('/')}/api/v1"
        self._timeout = timeout

    @staticmethod
    def _auth(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    async def _request(
        self, method: str, path: str, *, token: str, context: str, **kwargs: Any
    ) -> httpx.Response:
        url = f"{self._base}{path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.request(
                    method, url, headers=self._auth(token), **kwargs
                )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Catalogue request failed ({context}): {exc}") from exc
        _raise_for_status(resp, context=context)
        return resp

    async def get_project(self, project_id: str, token: str) -> dict | None:
        """Return the project if the caller may access it; None on 404.

        This single call doubles as the per-request access check and the
        lazy-JIT source. A 404 means "no access OR deleted" — both cases mean
        the editor must not expose the project.
        """
        try:
            resp = await self._request(
                "GET", f"/projects/{project_id}", token=token, context="get_project"
            )
        except NotFoundError:
            return None
        return resp.json()

    async def list_authorization(self, token: str) -> dict:
        """GET /me/authorization → {managedOrganisations, projectMemberships, ...}."""
        resp = await self._request(
            "GET", "/me/authorization", token=token, context="list_authorization"
        )
        return resp.json()

    async def list_org_projects(self, org_id: str, token: str) -> list[dict]:
        """Projects in an org visible to the caller (managers: all; curators: own)."""
        resp = await self._request(
            "GET",
            f"/organisations/{org_id}/projects",
            token=token,
            context="list_org_projects",
        )
        data = resp.json()
        return data if isinstance(data, list) else []

    async def list_organisations(self, token: str) -> list[dict]:
        """Organisations visible to the caller (used to resolve org names)."""
        resp = await self._request(
            "GET", "/organisations", token=token, context="list_organisations"
        )
        data = resp.json()
        return data if isinstance(data, list) else []

    async def create_project(
        self, org_id: str, name: str, slug: str, token: str
    ) -> dict:
        """POST /organisations/{orgId}/projects (create-up). Manager-of-org only."""
        resp = await self._request(
            "POST",
            f"/organisations/{org_id}/projects",
            token=token,
            context="create_project",
            json={"name": name, "slug": slug, "toolSlug": settings.tool_slug},
        )
        return resp.json()

    async def update_project_name(
        self, project_id: str, name: str, token: str
    ) -> dict:
        """PUT /projects/{id} — name only (slug/tool immutable). Manager-of-org only."""
        resp = await self._request(
            "PUT",
            f"/projects/{project_id}",
            token=token,
            context="update_project_name",
            json={"name": name},
        )
        return resp.json()

    async def delete_project(self, project_id: str, token: str) -> None:
        """DELETE /projects/{id} — soft delete. Manager-of-org only."""
        await self._request(
            "DELETE",
            f"/projects/{project_id}",
            token=token,
            context="delete_project",
        )


_client: CatalogueClient | None = None


def get_catalogue_client() -> CatalogueClient:
    """FastAPI dependency returning a process-wide CatalogueClient.

    Tests override this via ``app.dependency_overrides`` with an in-memory fake.
    """
    global _client
    if _client is None:
        if not settings.catalogue_base_url:
            raise AppError("CATALOGUE_BASE_URL is not configured")
        _client = CatalogueClient(
            settings.catalogue_base_url, settings.catalogue_timeout_seconds
        )
    return _client
