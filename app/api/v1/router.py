"""Aggregate v1 API routers."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.v1 import agent, assets, capabilities, jobs, projects
from app.core.security import get_current_principal

# Every /api/v1 route requires a valid Keycloak token. Project-level
# authorization is layered on top per-route via app.api.deps.
api_router = APIRouter(dependencies=[Depends(get_current_principal)])
api_router.include_router(projects.router)
api_router.include_router(assets.router)
api_router.include_router(capabilities.router)
api_router.include_router(jobs.router)
api_router.include_router(agent.router)
