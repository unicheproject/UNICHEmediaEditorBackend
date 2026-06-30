"""Project request/response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel

# Catalogue slug rule: lowercase alphanumeric/dashes, 2-63 chars.
SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{1,62}$"


class ProjectCreate(BaseModel):
    """Create-up payload: the editor creates the project in the catalogue.

    ``org_id`` selects which organisation to create under (the user must be a
    manager of it). ``description`` is stored only in the editor.
    """

    name: str = Field(min_length=1, max_length=200)
    slug: str = Field(pattern=SLUG_PATTERN)
    org_id: uuid.UUID
    description: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None


class ProjectRead(ORMModel):
    id: uuid.UUID
    org_id: uuid.UUID | None
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None


class ProjectListItem(BaseModel):
    """Picker entry — built live from the catalogue, merged with local description."""

    id: uuid.UUID
    org_id: uuid.UUID | None = None
    name: str
    slug: str | None = None
    status: str | None = None
    description: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
