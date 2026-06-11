"""Small helpers shared by local-tool handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.capabilities.context import JobContext
from app.core.errors import ValidationError


def require_input(ctx: JobContext) -> str:
    if not ctx.input_path:
        raise ValidationError(f"Capability '{ctx.capability_id}' requires an input asset")
    return ctx.input_path


def require(params: dict[str, Any], key: str) -> Any:
    if key not in params or params[key] is None:
        raise ValidationError(f"Missing required parameter '{key}'")
    return params[key]


def input_stem(ctx: JobContext) -> str:
    name = ctx.input_asset_meta.get("original_filename") or (
        Path(ctx.input_path).name if ctx.input_path else "output"
    )
    return Path(name).stem or "output"


def input_ext(ctx: JobContext) -> str:
    if ctx.input_path:
        return Path(ctx.input_path).suffix.lstrip(".").lower() or "bin"
    return "bin"
