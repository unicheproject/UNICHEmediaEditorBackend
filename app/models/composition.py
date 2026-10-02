"""Project-owned composition heads and immutable document revisions."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JSONType, TimestampMixin


class ProjectComposition(Base, TimestampMixin):
    __tablename__ = "project_compositions"
    __table_args__ = (UniqueConstraint("project_id", name="uq_project_compositions_project_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id"), nullable=False
    )
    current_revision_number: Mapped[int] = mapped_column(Integer, nullable=False)


class ProjectCompositionRevision(Base):
    __tablename__ = "project_composition_revisions"
    __table_args__ = (
        UniqueConstraint(
            "composition_id", "revision_number", name="uq_composition_revision_number"
        ),
        CheckConstraint("revision_number > 0", name="ck_composition_revision_positive"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    composition_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("project_compositions.id"), nullable=False, index=True
    )
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    document: Mapped[dict] = mapped_column(JSONType, nullable=False)
    document_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    author_subject: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
