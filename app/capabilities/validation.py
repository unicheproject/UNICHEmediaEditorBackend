"""Validation helpers for the canonical capability input contract."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from app.schemas.capability import CapabilityMetadata


class CapabilityInputValidationError(ValueError):
    """A controlled, consumer-neutral capability input validation failure."""


@dataclass(frozen=True)
class MediaAssetReference:
    """A literal asset reference and the semantic contract attached to it."""

    parameter: str
    asset_id: str
    role: str
    media_types: tuple[str, ...]


def _format_checker(*, allow_step_refs: bool) -> FormatChecker:
    checker = FormatChecker()

    @checker.checks("uuid")
    def _is_uuid(value: object) -> bool:
        if not isinstance(value, str):
            return True  # JSON Schema's type keyword reports the type failure.
        if allow_step_refs and value.startswith("@"):
            return True
        try:
            uuid.UUID(value)
        except (ValueError, AttributeError):
            return False
        return True

    return checker


def validate_capability_input(
    capability: CapabilityMetadata,
    value: Mapping[str, Any],
    *,
    allow_step_refs: bool = False,
) -> None:
    """Validate input with the registry schema, including active UUID formats."""
    validator = Draft202012Validator(
        capability.input_schema,
        format_checker=_format_checker(allow_step_refs=allow_step_refs),
    )
    errors = sorted(validator.iter_errors(dict(value)), key=lambda error: list(error.path))
    if not errors:
        return

    error: JsonSchemaValidationError = errors[0]
    path = ".".join(str(part) for part in error.absolute_path)
    location = f"input.{path}" if path else "input"
    raise CapabilityInputValidationError(f"{location}: {error.message}")


def media_asset_semantics(schema: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Read item-level Asset semantics, retaining legacy array annotations."""
    reference_schema = schema.get("items", {}) if schema.get("type") == "array" else schema
    semantics: Mapping[str, Any] | None = (
        reference_schema.get("x-media-asset") or schema.get("x-media-asset")
    )
    return semantics


def iter_media_asset_references(
    capability: CapabilityMetadata, value: Mapping[str, Any]
) -> Iterator[MediaAssetReference]:
    """Yield references declared by the schema's shared x-media-asset semantics."""
    properties = capability.input_schema.get("properties", {})
    for parameter, property_schema in properties.items():
        semantics = media_asset_semantics(property_schema)
        if not semantics or parameter not in value or value[parameter] in (None, ""):
            continue
        raw_values = (
            value[parameter] if property_schema.get("type") == "array" else [value[parameter]]
        )
        for raw_value in raw_values:
            yield MediaAssetReference(
                parameter=parameter,
                asset_id=str(raw_value),
                role=str(semantics["role"]),
                media_types=tuple(str(item) for item in semantics.get("media_types", [])),
            )
