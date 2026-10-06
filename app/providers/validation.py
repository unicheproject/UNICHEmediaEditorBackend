"""Small provider-neutral success contracts; never echo provider diagnostics."""

from __future__ import annotations

from typing import Any

from app.core.errors import ProviderError


def require_success_object(payload: object) -> dict[str, Any]:
    """Failure wins: error/failed/not_implemented, false success/ok, or truthy error.

    Unknown status values and message/detail fields are not failure markers.
    """
    if not isinstance(payload, dict):
        raise ProviderError("Inference provider returned non-object JSON")
    status = payload.get("status")
    if (
        (
            isinstance(status, str)
            and status.strip().lower()
            in {
                "error",
                "failed",
                "not_implemented",
            }
        )
        or payload.get("success") is False
        or payload.get("ok") is False
        or payload.get("error")
    ):
        raise ProviderError("Inference provider reported failure")
    return payload


def require_caption(payload: object) -> dict[str, Any]:
    result = require_success_object(payload)
    caption = result.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ProviderError("Inference provider returned no valid caption")
    return result


def require_transcript(payload: object) -> dict[str, Any]:
    result = require_success_object(payload)
    # Explicitly empty text represents silence; missing text never does.
    if "text" not in result or not isinstance(result["text"], str):
        raise ProviderError("Inference provider returned no valid transcript text")
    return result
