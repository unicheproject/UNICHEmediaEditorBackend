"""Job request/response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import JobStatus
from app.schemas.common import ORMModel


class JobCreate(BaseModel):
    capability_id: str = Field(min_length=1, max_length=128)
    project_id: uuid.UUID | None = None
    asset_id: uuid.UUID | None = None
    input: dict[str, Any] = Field(default_factory=dict)


class JobRead(ORMModel):
    id: uuid.UUID
    project_id: uuid.UUID | None
    asset_id: uuid.UUID | None
    capability_id: str
    status: JobStatus
    input: dict[str, Any]
    output: dict[str, Any] | None
    error: str | None
    progress: int
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
