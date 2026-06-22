"""Plan validation against the capability registry.

The backend (mock or OpenRouter) proposes a PlannerResult; this module checks a
Plan is executable before it's shown to the user: capabilities exist + enabled,
params satisfy each capability's JSON input_schema, asset references resolve,
and primary-asset media types match.
"""

from __future__ import annotations

from typing import Any

import jsonschema

from app.agent.schemas import Plan, PlanStep, is_step_ref, step_ref_id
from app.capabilities import registry
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
    candidates = [step.asset, *step.assets, *step.params.values()]
    for value in candidates:
        if is_step_ref(value):
            refs.append(step_ref_id(value))  # type: ignore[arg-type]
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
        try:
            jsonschema.validate(_effective_input(step), cap.input_schema)
        except jsonschema.ValidationError as exc:
            raise PlanValidationError(
                f"Step '{step.id}' ({step.capability_id}) invalid params: {exc.message}"
            ) from exc

        # references: '@step' must point at an earlier step; literals must be in scope
        literals = [a for a in [step.asset, *step.assets] if a and not is_step_ref(a)]
        for ref in _refs_in_step(step):
            if ref not in seen_steps:
                raise PlanValidationError(
                    f"Step '{step.id}' references unknown/earlier-undefined step '{ref}'."
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
