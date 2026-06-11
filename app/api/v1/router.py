"""Aggregate v1 API routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import assets, capabilities, jobs, projects

api_router = APIRouter()
api_router.include_router(projects.router)
api_router.include_router(assets.router)
api_router.include_router(capabilities.router)
api_router.include_router(jobs.router)
