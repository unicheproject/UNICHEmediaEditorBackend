"""Project ORM model."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.asset import Asset


class Project(Base, TimestampMixin, SoftDeleteMixin):
    """Companion row for a Catalogue project.

    ``id`` is the Catalogue's project UUID (the canonical platform id) — the
    editor never mints its own. ``description`` is editor-local only (the
    catalogue has no description field). Provisioned lazily on first authorized
    access, or on create-up.
    """

    __tablename__ = "projects"

    # PK == Catalogue project UUID. No client-side default: the id is always
    # supplied (from the catalogue) on insert.
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    org_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    assets: Mapped[list[Asset]] = relationship(
        back_populates="project", lazy="noload"
    )
