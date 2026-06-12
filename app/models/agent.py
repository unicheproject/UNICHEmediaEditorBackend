"""Agent conversation + plan ORM models."""

from __future__ import annotations

import uuid

from sqlalchemy import Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JSONType, TimestampMixin
from app.models.enums import AgentPlanStatus


class AgentSession(Base, TimestampMixin):
    __tablename__ = "agent_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )
    # In-scope asset ids (list of UUID strings).
    asset_ids: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)
    # Full conversation transcript, including assistant reasoning_details.
    messages: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)


class AgentPlan(Base, TimestampMixin):
    __tablename__ = "agent_plans"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agent_sessions.id"), nullable=False, index=True
    )
    summary: Mapped[str] = mapped_column(String(1024), default="")
    status: Mapped[AgentPlanStatus] = mapped_column(
        Enum(AgentPlanStatus, name="agent_plan_status"),
        nullable=False,
        default=AgentPlanStatus.proposed,
    )
    # Proposed steps (as planned) and per-step run state (job_id/status/output).
    steps: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)
    step_runs: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)
    result_asset_ids: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
