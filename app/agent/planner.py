"""Plan validation against the capability registry.

The backend (mock or OpenRouter) proposes a PlannerResult; this module checks a
Plan is executable before it's shown to the user: capabilities exist + enabled,
params satisfy each capability's JSON input_schema, asset references resolve,
and primary-asset media types match.
"""

from __future__ import annotations

from typing import Any

from app.agent.schemas import Plan, PlanStep, is_step_ref, step_ref_id
from app.capabilities import registry
from app.capabilities.validation import (
    CapabilityInputValidationError,
    iter_media_asset_references,
    validate_capability_input,
)
from app.core.errors import NotFoundError
from app.models.enums import CostClass


class PlanValidationError(Exception):
    """Raised with a model-readable message so the planner can request a repair."""


def _effective_input(step: PlanStep) -> dict[str, Any]:
    """Reconstruct the job.input the executor will build, for schema validation."""
    eff = dict(step.params)
    if step.assets:
        eff.setdefault("asset_ids", list(step.assets))
    return eff


def _refs_in_step(step: PlanStep) -> list[str]:
    refs: list[str] = []
    def visit(value: Any) -> None:
        if is_step_ref(value):
            refs.append(step_ref_id(value))  # type: ignore[arg-type]
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for value in [step.asset, *step.assets, *step.params.values()]:
        visit(value)
    return refs


def validate_plan(
    plan: Plan, assets_by_id: dict[str, str], *, deterministic_only: bool = False
) -> None:
    """Validate a plan. `assets_by_id` maps in-scope asset id -> media_type.

    When `deterministic_only` is set, any hosted-AI / GPU capability is rejected
    with a repairable message (defence-in-depth alongside the filtered catalog).

    Raises PlanValidationError with a concrete message on the first problem.
    """
    if not plan.steps:
        raise PlanValidationError("Plan has no steps.")

    seen_steps: set[str] = set()
    capability_by_step: dict[str, str] = {}
    for step in plan.steps:
        # capability exists + enabled
        try:
            cap = registry.get(step.capability_id)
        except NotFoundError as exc:
            raise PlanValidationError(str(exc)) from exc
        if not cap.enabled:
            raise PlanValidationError(f"Capability '{step.capability_id}' is disabled.")
        if deterministic_only and cap.cost_class != CostClass.deterministic:
            raise PlanValidationError(
                f"Capability '{step.capability_id}' is not available to the agent; "
                "only deterministic (local-tool) capabilities may be used."
            )

        # params + derived asset_ids validate against the capability schema
        effective_input = _effective_input(step)
        try:
            validate_capability_input(cap, effective_input, allow_step_refs=True)
        except CapabilityInputValidationError as exc:
            raise PlanValidationError(
                f"Step '{step.id}' ({step.capability_id}) invalid params: {exc}"
            ) from exc

        # references: '@step' must point at an earlier step; literals must be in scope
        literals = [a for a in [step.asset, *step.assets] if a and not is_step_ref(a)]
        for ref in _refs_in_step(step):
            if ref not in seen_steps:
                raise PlanValidationError(
                    f"Step '{step.id}' references unknown/earlier-undefined step '{ref}'."
                )
            if capability_by_step.get(ref) == "video.split":
                raise PlanValidationError(
                    f"Step '{step.id}' cannot use '@{ref}' as one asset: video.split "
                    "returns multiple segments. Select a segment explicitly in a future flow."
                )
        for dep in step.depends_on:
            if dep not in seen_steps:
                raise PlanValidationError(
                    f"Step '{step.id}' depends_on undefined step '{dep}'."
                )
        for lit in literals:
            if lit not in assets_by_id:
                raise PlanValidationError(
                    f"Step '{step.id}' references asset '{lit}' not in this session's scope."
                )

        # Literal named/list asset parameters must be in scope and match the
        # media category declared by the canonical capability schema.
        for reference in iter_media_asset_references(cap, effective_input):
            if is_step_ref(reference.asset_id):
                continue
            media = assets_by_id.get(reference.asset_id)
            if media is None:
                raise PlanValidationError(
                    f"Step '{step.id}' references asset '{reference.asset_id}' "
                    "not in this session's scope."
                )
            if reference.media_types and media not in reference.media_types:
                raise PlanValidationError(
                    f"Step '{step.id}' parameter '{reference.parameter}' needs "
                    f"{list(reference.media_types)} media, but asset "
                    f"'{reference.asset_id}' is {media}."
                )

        # primary-asset media type must be supported (only for literal primaries)
        if step.asset and not is_step_ref(step.asset):
            media = assets_by_id.get(step.asset)
            supported = [m.value for m in cap.supported_media_types]
            if supported and media not in supported:
                raise PlanValidationError(
                    f"Step '{step.id}' ({step.capability_id}) needs {supported} input, "
                    f"but asset '{step.asset}' is {media}."
                )

        seen_steps.add(step.id)
        capability_by_step[step.id] = step.capability_id
