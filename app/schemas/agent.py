"""Agent request/response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.enums import AgentPlanStatus
from app.schemas.common import ORMModel


class AgentSessionCreate(BaseModel):
    project_id: uuid.UUID
    asset_ids: list[uuid.UUID] = Field(default_factory=list)


class TranscriptMessage(BaseModel):
    role: str
    content: str | None


class AgentSessionRead(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    asset_ids: list[uuid.UUID]
    transcript: list[TranscriptMessage]
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    content: str = Field(min_length=1)


class AgentPlanRead(ORMModel):
    id: uuid.UUID
    session_id: uuid.UUID
    summary: str
    status: AgentPlanStatus
    steps: list[dict[str, Any]]
    step_runs: list[dict[str, Any]]
    result_asset_ids: list[uuid.UUID]
    error: str | None
    created_at: datetime
    updated_at: datetime


class AgentMessageResponse(BaseModel):
    type: Literal["plan", "clarification"]
    # clarification fields
    question: str | None = None
    missing: list[str] = Field(default_factory=list)
    # plan field
    plan: AgentPlanRead | None = None
