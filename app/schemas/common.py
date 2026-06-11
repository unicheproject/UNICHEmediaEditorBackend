"""Shared schema utilities."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# Pagination defaults, reused across list endpoints.
DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200


class ORMModel(BaseModel):
    """Base for response models read from ORM objects."""

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str


class Page[T](BaseModel):
    """A paginated slice of a collection."""

    items: list[T]
    total: int
    limit: int
    offset: int
