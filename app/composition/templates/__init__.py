"""Built-in, versioned composition templates (MED/028) and their compiler.

Templates are JSON files shipped with the code, listed explicitly below; a
user-supplied template ID is only ever a key into this registry, never a path.
Published versions never change: a change is a new ``vN.json`` (tests pin the
published files' hashes). Every file is validated when the registry loads, which
the API router triggers at import, so an invalid bundled template fails startup
instead of being skipped.
"""

from __future__ import annotations

import json
import uuid
from functools import cache
from importlib import resources
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.core.errors import ValidationError
from app.schemas.composition import CompositionDocument
from app.schemas.composition_template import CompositionTemplate

# (template_id, version, file relative to this package). Append new versions;
# never edit or remove a published one.
TEMPLATE_FILES: tuple[tuple[str, int, str], ...] = (
    ("exhibit-intro", 1, "exhibit-intro/v1.json"),
    ("artefact-spotlight", 1, "artefact-spotlight/v1.json"),
    ("event-recap", 1, "event-recap/v1.json"),
    ("educational-explainer", 1, "educational-explainer/v1.json"),
)

# Generated track and clip IDs are UUIDv5 names under this fixed namespace, so
# one template version with the same ordered bindings yields the same document.
TEMPLATE_ID_NAMESPACE = uuid.UUID("6f1c2b6e-5d0a-4c55-9a51-2f3e8d0b7c41")

Bindings = dict[str, uuid.UUID | list[uuid.UUID] | None]


class TemplateDefinitionError(ValueError):
    """A bundled template file is invalid (raised at import, never at request time)."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise TemplateDefinitionError("duplicate JSON keys")
    return dict(pairs)


def parse_template(raw: str | bytes) -> CompositionTemplate:
    """Strictly parse and check one template definition."""
    try:
        json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        template = CompositionTemplate.model_validate_json(raw)
    except (PydanticValidationError, ValueError) as exc:
        raise TemplateDefinitionError(str(exc)) from exc
    _check_suggestions(template)
    for bindings in (_dry_bindings(template, required_only=True),
                     _dry_bindings(template, required_only=False)):
        try:
            compile_template(template, bindings)
        except ValidationError as exc:
            raise TemplateDefinitionError(
                f"{template.template_id} v{template.version}: {exc.message}"
            ) from exc
    return template


def _check_suggestions(template: CompositionTemplate) -> None:
    from jsonschema import Draft202012Validator

    from app.capabilities import registry

    for name, slot in template.slots.items():
        if slot.suggestion is None:
            continue
        schema = dict(registry.get(slot.suggestion.capability_id).input_schema)
        unknown = set(slot.suggestion.params) - set(schema.get("properties", {}))
        schema.pop("required", None)
        errors = list(Draft202012Validator(schema).iter_errors(slot.suggestion.params))
        if unknown or errors:
            raise TemplateDefinitionError(f"slot {name!r} has an invalid suggestion")


def _dry_bindings(template: CompositionTemplate, *, required_only: bool) -> Bindings:
    bindings: Bindings = {}
    for index, (name, slot) in enumerate(template.slots.items()):
        if required_only and not slot.required:
            continue
        ids = [uuid.uuid5(TEMPLATE_ID_NAMESPACE, f"dry:{index}:{n}")
               for n in range(1 if required_only else slot.max_assets)]
        bindings[name] = ids if slot.multiple else ids[0]
    return bindings


def _resolve_bindings(
    template: CompositionTemplate, bindings: Bindings
) -> dict[str, list[uuid.UUID]]:
    unknown = sorted(set(bindings) - set(template.slots))
    if unknown:
        raise ValidationError(f"Unknown template slot(s): {', '.join(unknown)}")
    resolved: dict[str, list[uuid.UUID]] = {}
    for name, slot in template.slots.items():
        value = bindings.get(name)
        if value is None:
            if slot.required:
                raise ValidationError(f"Template slot {name!r} is required")
            resolved[name] = []
            continue
        if slot.multiple != isinstance(value, list):
            shape = "a list of Asset IDs" if slot.multiple else "a single Asset ID"
            raise ValidationError(f"Template slot {name!r} takes {shape}")
        ids = value if isinstance(value, list) else [value]
        if not ids and slot.required:
            raise ValidationError(f"Template slot {name!r} needs at least one Asset")
        if len(ids) > slot.max_assets:
            raise ValidationError(
                f"Template slot {name!r} takes at most {slot.max_assets} Assets"
            )
        resolved[name] = list(ids)
    return resolved


def compile_template(template: CompositionTemplate, bindings: Bindings) -> CompositionDocument:
    """Bind slots to Assets and return an ordinary, validated composition document."""
    assets = _resolve_bindings(template, bindings)
    prefix = f"uniche-template:{template.template_id}:v{template.version}"

    def stable_id(*parts: object) -> str:
        return str(uuid.uuid5(TEMPLATE_ID_NAMESPACE, ":".join([prefix, *map(str, parts)])))

    tracks: list[dict[str, Any]] = []
    for t, track in enumerate(template.tracks):
        clips: list[dict[str, Any]] = []
        for c, clip in enumerate(track.clips):
            fields = clip.model_dump(exclude={"slot"}, exclude_none=True)
            for k, asset_id in enumerate(assets[clip.slot]):
                clip_id = stable_id("track", t, "clip", c, "slot", clip.slot, "occurrence", k)
                clips.append({"id": clip_id, "asset_id": str(asset_id), **fields})
        if clips:
            tracks.append({"id": stable_id("track", t), "kind": track.kind, "clips": clips})
        elif t == 0:
            raise ValidationError("The template's video track has no clips to show")
    bed = template.audio_bed
    bed_assets = assets[bed.slot] if bed is not None else []
    document = {
        "schema_version": 1,
        "canvas": template.canvas.model_dump(),
        "tracks": tracks,
        "audio_bed": (
            {"asset_id": str(bed_assets[0]), "mode": bed.mode.value, "volume": bed.volume}
            if bed is not None and bed_assets else None
        ),
    }
    try:
        return CompositionDocument.model_validate(document)
    except PydanticValidationError as exc:
        raise ValidationError("The template produced an invalid composition") from exc


@cache
def _registry() -> dict[tuple[str, int], CompositionTemplate]:
    templates: dict[tuple[str, int], CompositionTemplate] = {}
    root = resources.files(__name__)
    for template_id, version, filename in TEMPLATE_FILES:
        raw = root.joinpath(filename).read_bytes()
        template = parse_template(raw)
        if (template.template_id, template.version) != (template_id, version):
            raise TemplateDefinitionError(f"{filename} does not declare {template_id} v{version}")
        if (template_id, version) in templates:
            raise TemplateDefinitionError(f"{template_id} v{version} is listed twice")
        templates[(template_id, version)] = template
    return templates


def get_template(template_id: str, version: int) -> CompositionTemplate | None:
    return _registry().get((template_id, version))


def latest_template(template_id: str) -> CompositionTemplate | None:
    versions = [t for (tid, _), t in _registry().items() if tid == template_id]
    return max(versions, key=lambda t: t.version) if versions else None


def latest_templates() -> list[CompositionTemplate]:
    ids = dict.fromkeys(template_id for template_id, _, _ in TEMPLATE_FILES)
    return [t for t in (latest_template(i) for i in ids) if t is not None]


def load_registry() -> None:
    """Load and validate every bundled template now (called at API import)."""
    _registry()
