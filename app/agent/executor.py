"""Execute an approved agent plan as a chained sequence of capability jobs.

Reuses services.jobs (create_job + execute_job) so every step is a real,
tracked Job with media-type validation, derived-asset creation and status — and
each step's output asset feeds dependent steps via '@stepN' references.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import Plan, PlanStep, is_step_ref, step_ref_id
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.agent import AgentPlan, AgentSession
from app.models.enums import AgentPlanStatus, JobStatus
from app.schemas.job import JobCreate
from app.services import jobs as jobs_svc

logger = get_logger(__name__)


def _resolve(ref: str, outputs: dict[str, str | None]) -> str:
    if is_step_ref(ref):
        sid = step_ref_id(ref)
        out = outputs.get(sid)
        if not out:
            raise ValueError(f"Step '{sid}' produced no asset to reference")
        return out
    return ref


def _resolve_params(params: dict[str, Any], outputs: dict[str, str | None]) -> dict[str, Any]:
    resolved = {}
    for key, value in params.items():
        if key == "asset_ids":
            continue  # rebuilt from step.assets
        resolved[key] = _resolve(value, outputs) if is_step_ref(value) else value
    return resolved


def _build_job_input(
    step: PlanStep, outputs: dict[str, str | None]
) -> tuple[uuid.UUID | None, dict]:
    primary = _resolve(step.asset, outputs) if step.asset else None
    job_input = _resolve_params(step.params, outputs)
    if step.assets:
        job_input["asset_ids"] = [_resolve(a, outputs) for a in step.assets]
    asset_id = uuid.UUID(primary) if primary else None
    return asset_id, job_input


async def execute_plan(session: AsyncSession, plan_id: uuid.UUID) -> AgentPlan:
    plan_row = await session.get(AgentPlan, plan_id)
    if plan_row is None:
        raise ValueError(f"AgentPlan '{plan_id}' not found")
    agent_session = await session.get(AgentSession, plan_row.session_id)
    if agent_session is None:
        raise NotFoundError(f"Agent session '{plan_row.session_id}' not found")
    project_id = agent_session.project_id

    plan = Plan.model_validate(
        {"type": "plan", "summary": plan_row.summary, "steps": plan_row.steps}
    )
    plan_row.status = AgentPlanStatus.running
    await session.commit()

    outputs: dict[str, str | None] = {}
    step_runs: list[dict] = []
    try:
        for step in plan.steps:
            asset_id, job_input = _build_job_input(step, outputs)
            job = await jobs_svc.create_job(
                session,
                JobCreate(
                    capability_id=step.capability_id,
                    project_id=project_id,
                    asset_id=asset_id,
                    input=job_input,
                ),
            )
            job = await jobs_svc.execute_job(session, job.id)
            out_asset = None
            if job.output and job.output.get("outputs"):
                out_asset = job.output["outputs"][0]["asset_id"]
            outputs[step.id] = out_asset
            step_runs.append({
                "step_id": step.id,
                "capability_id": step.capability_id,
                "job_id": str(job.id),
                "status": job.status.value,
                "output_asset_id": out_asset,
            })
            plan_row.step_runs = list(step_runs)
            await session.commit()
            if job.status == JobStatus.failed:
                raise RuntimeError(f"Step '{step.id}' failed: {job.error}")

        plan_row.status = AgentPlanStatus.succeeded
        last = step_runs[-1]["output_asset_id"] if step_runs else None
        plan_row.result_asset_ids = [a for a in [last] if a]
    except Exception as exc:  # noqa: BLE001 - record failure on the plan
        logger.exception("Plan %s failed", plan_id)
        plan_row.status = AgentPlanStatus.failed
        plan_row.error = str(exc)
    finally:
        plan_row.step_runs = list(step_runs)
        await session.commit()

    await session.refresh(plan_row)
    return plan_row
