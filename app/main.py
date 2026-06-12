"""FastAPI application entrypoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.v1 import health
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="MVP backend for the UNICHE Media Editor.",
)

# CORS — wide open by default for local frontend development. Note: when origins
# is "*", credentials must be disabled (browsers reject "*" + credentials).
_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

# Health is mounted at the root (GET /health); everything else under /api/v1.
app.include_router(health.router)
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> str:
    placeholder = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if placeholder.exists():
        return placeholder.read_text(encoding="utf-8")
    return "<h1>UNICHE Media Editor API</h1><p>See <a href='/docs'>/docs</a>.</p>"
