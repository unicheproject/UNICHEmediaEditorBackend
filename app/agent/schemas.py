"""Agent planning types.

A planner turns a user message into either a clarifying question or a Plan: an
ordered list of capability invocations.

Asset references are strings: a literal uploaded asset id (UUID), or a
prior-step reference of the form ``@<step_id>`` meaning "the asset produced by
that step" (resolved at execution time).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

STEP_REF_PREFIX = "@"


def is_step_ref(value: object) -> bool:
    return isinstance(value, str) and value.startswith(STEP_REF_PREFIX)


def step_ref_id(value: str) -> str:
    return value[len(STEP_REF_PREFIX):]


class PlanStep(BaseModel):
    id: str = Field(description="Unique step id within the plan, e.g. 'step1'.")
    capability_id: str
    # Capability params (validated against the capability input_schema).
    params: dict[str, Any] = Field(default_factory=dict)
    # Primary input: an uploaded asset id or '@<step_id>'.
    asset: str | None = None
    # Extra ordered inputs for multi-input ops (concat/compose/slideshow).
    assets: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    type: Literal["plan"] = "plan"
    summary: str = ""
    steps: list[PlanStep]


class Clarification(BaseModel):
    type: Literal["clarification"] = "clarification"
    question: str
    missing: list[str] = Field(default_factory=list)


# The planner backend returns one of these (discriminated on `type`).
PlannerResult = Plan | Clarification
