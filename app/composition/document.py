"""Canonical serialization helpers for validated composition documents."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.schemas.composition import CompositionDocument


def canonical_document(document: CompositionDocument) -> tuple[dict[str, Any], str]:
    """Return normalized JSON data and its deterministic SHA-256 identity."""
    data = document.model_dump(mode="json")
    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return data, hashlib.sha256(payload).hexdigest()
