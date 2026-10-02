"""Job ORM model."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JSONType, TimestampMixin
from app.models.enums import JobStatus


class Job(Base, TimestampMixin):
    __tablename__ = "jobs"
    __table_args__ = (
        Index(
            "ix_jobs_agent_reuse_lookup",
            "agent_plan_id",
            "agent_attempt_id",
            "execution_identity",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("projects.id"), nullable=True, index=True
    )
    asset_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("assets.id"), nullable=True, index=True
    )
    capability_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # Internal provenance for deterministic same-AgentPlan retry reuse.
    # These are NULL for direct Jobs and for executions without a complete,
    # reusable single-Asset output. They are intentionally not part of JobRead.
    agent_plan_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agent_plans.id", ondelete="SET NULL"), nullable=True
    )
    agent_attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agent_plan_attempts.id", ondelete="SET NULL"), nullable=True
    )
    execution_identity: Mapped[str | None] = mapped_column(String(64), nullable=True)

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"),
        nullable=False,
        default=JobStatus.queued,
    )
    input: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    output: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
