"""Agent conversation + plan ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, UniqueConstraint
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
    # Compiled Agent execution schedule, count summary, and source fingerprints.
    # Nullable for plans persisted before the render optimizer was introduced.
    execution_summary: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    result_asset_ids: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Supplied to ARQ and checked with a DB compare-and-set by the worker.
    # A stale delivery cannot claim a later reapproval of this plan.
    active_attempt_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)


class AgentPlanAttempt(Base, TimestampMixin):
    """One physical execution attempt of a logical AgentPlan."""

    __tablename__ = "agent_plan_attempts"
    __table_args__ = (
        UniqueConstraint("plan_id", "attempt_number", name="uq_agent_plan_attempt_number"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    plan_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agent_plans.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="approved")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    step_runs: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)
    execution_summary: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    job_ids: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)
    result_asset_ids: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
