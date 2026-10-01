"""Execute an approved agent plan as a chained sequence of capability jobs.

Reuses services.jobs (create_job + execute_job) so every step is a real,
tracked Job with media-type validation, derived-asset creation and status — and
each step's output asset feeds dependent steps via '@stepN' references.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.agent import render as render_compiler
from app.agent.schemas import Plan, PlanStep, is_step_ref, step_ref_id
from app.capabilities.context import HandlerResult, JobContext, OutputFile
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.models.agent import AgentPlan, AgentSession
from app.models.asset import Asset
from app.models.enums import AgentPlanStatus, JobStatus, MediaType
from app.models.job import Job
from app.schemas.job import JobCreate
from app.services import jobs as jobs_svc
from app.services.assets import get_asset
from app.services.storage import get_storage

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
    # Multi-input ops take their ordered inputs from `step.assets`; tolerate
    # planners that instead place them in `params.asset_ids`.
    raw_ids = step.assets
    if not raw_ids and isinstance(step.params.get("asset_ids"), list):
        raw_ids = step.params["asset_ids"]
    if raw_ids:
        job_input["asset_ids"] = [_resolve(a, outputs) for a in raw_ids]
    asset_id = uuid.UUID(primary) if primary else None
    return asset_id, job_input


def _consumed_step_ids(plan: Plan) -> set[str]:
    """Step ids whose output is referenced (@stepN) by another step.

    Their produced assets are intermediate; the rest are plan deliverables.
    """
    consumed: set[str] = set()
    def visit(value: Any) -> None:
        if is_step_ref(value):
            consumed.add(step_ref_id(value))  # type: ignore[arg-type]
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for step in plan.steps:
        for value in [step.asset, *step.assets, *step.params.values()]:
            visit(value)
    return consumed


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
    summary = plan_row.execution_summary
    if summary is not None:
        if summary.get("optimizer_version") != render_compiler.OPTIMIZER_VERSION:
            plan_row.status = AgentPlanStatus.failed
            plan_row.error = "The approved render plan is stale; please request a new plan."
            await session.commit()
            return plan_row
        if summary.get("input_fingerprint") != render_compiler.plan_fingerprint(plan):
            plan_row.status = AgentPlanStatus.failed
            plan_row.error = "The approved plan inputs changed; please request a new plan."
            await session.commit()
            return plan_row
        try:
            await _validate_render_sources(session, summary, project_id)
        except (NotFoundError, ValidationError) as exc:
            plan_row.status = AgentPlanStatus.failed
            plan_row.error = str(exc)
            await session.commit()
            return plan_row
    plan_row.status = AgentPlanStatus.running
    await session.commit()

    outputs: dict[str, str | None] = {}
    step_runs: list[dict] = []
    try:
        if summary is None:
            # Plans persisted before the additive optimizer column retain their
            # original per-capability execution behavior.
            await _execute_legacy_steps(session, plan, project_id, outputs, step_runs)
        else:
            steps_by_id = {step.id: step for step in plan.steps}
            for node in summary["nodes"]:
                if node["kind"] == "identity":
                    for step_id in node["step_ids"]:
                        outputs[step_id] = node["output_asset_id"]
                        step = steps_by_id[step_id]
                        _update_summary_step(
                            summary, step_id, status="identity", identity=True,
                            job_id=None, output_asset_id=None,
                        )
                        step_runs.append({
                            "step_id": step_id, "capability_id": step.capability_id,
                            "status": "identity", "identity": True,
                            "render_group_id": node["id"], "job_id": None,
                            "output_asset_id": None,
                        })
                    plan_row.execution_summary = copy.deepcopy(summary)
                    flag_modified(plan_row, "execution_summary")
                    plan_row.step_runs = list(step_runs)
                    await session.commit()
                    continue
                if node["kind"] == "render_group":
                    job, output_asset_id = await _execute_render_group(
                        session, plan_row, project_id, node
                    )
                    node["job_id"] = str(job.id)
                    node["status"] = job.status.value
                    node["output_asset_id"] = output_asset_id
                    active_entries = [op for op in node["operations"] if not op["identity"]]
                    terminal_active = active_entries[-1]["step_id"] if active_entries else None
                    for step_id in node["step_ids"]:
                        step = steps_by_id[step_id]
                        entry = next(op for op in node["operations"] if op["step_id"] == step_id)
                        is_identity = bool(entry["identity"])
                        status = "identity" if is_identity else job.status.value
                        step_runs.append({
                            "step_id": step_id, "capability_id": step.capability_id,
                            "status": status, "identity": is_identity,
                            "render_group_id": node["id"], "job_id": str(job.id),
                            "output_asset_id": (
                                output_asset_id if step_id == terminal_active else None
                            ),
                        })
                        outputs[step_id] = output_asset_id
                        _update_summary_step(
                            summary, step_id, status=status, identity=is_identity,
                            job_id=str(job.id), output_asset_id=(
                                output_asset_id if step_id == terminal_active else None
                            ),
                        )
                    plan_row.execution_summary = copy.deepcopy(summary)
                    flag_modified(plan_row, "execution_summary")
                    plan_row.step_runs = list(step_runs)
                    await session.commit()
                    if job.status == JobStatus.failed:
                        raise RuntimeError(f"Render group '{node['id']}' failed: {job.error}")
                    continue

                step = steps_by_id[node["step_id"]]
                await _execute_standard_step(
                    session, project_id, step, outputs, step_runs, plan_row, summary
                )

        # Outputs consumed by a later step are intermediate; leaves are finals.
        consumed = _consumed_step_ids(plan)
        finals: list[str] = []
        for run in step_runs:
            if run.get("render_group_id"):
                continue
            out_assets = run.get("output_asset_ids") or ([run["output_asset_id"]]
                         if run.get("output_asset_id") else [])
            for out_asset in out_assets:
                if run["step_id"] in consumed:
                    asset = await session.get(Asset, uuid.UUID(out_asset))
                    if asset is not None:
                        asset.is_intermediate = True
                elif out_asset not in finals:
                    finals.append(out_asset)
        for node in (summary or {}).get("nodes", []):
            if node.get("kind") != "render_group" or not node.get("output_asset_id"):
                continue
            terminal_step = node["step_ids"][-1]
            asset_id = node["output_asset_id"]
            if terminal_step in consumed:
                asset = await session.get(Asset, uuid.UUID(asset_id))
                if asset is not None:
                    asset.is_intermediate = True
            elif asset_id not in finals:
                finals.append(asset_id)
        # Identity-only leaf steps return their unchanged source without creating
        # a Job or a derived Asset.
        for node in (summary or {}).get("nodes", []):
            if (
                node.get("kind") == "identity"
                and any(step_id not in consumed for step_id in node["step_ids"])
                and node["output_asset_id"] not in finals
            ):
                finals.append(node["output_asset_id"])
        plan_row.status = AgentPlanStatus.succeeded
        plan_row.result_asset_ids = finals
    except Exception as exc:  # noqa: BLE001 - record failure on the plan
        logger.exception("Plan %s failed", plan_id)
        plan_row.status = AgentPlanStatus.failed
        plan_row.error = str(exc)
    finally:
        plan_row.step_runs = list(step_runs)
        await session.commit()

    await session.refresh(plan_row)
    return plan_row


async def _validate_render_sources(
    session: AsyncSession, summary: dict, project_id: uuid.UUID
) -> None:
    """Reject stale/moved source inputs rather than silently changing a plan."""
    for node in summary.get("nodes", []):
        if node.get("kind") not in {"render_group", "identity"}:
            continue
        asset = await get_asset(session, uuid.UUID(node["source_asset_id"]))
        if asset.project_id != project_id:
            raise ValidationError("The approved source Asset is no longer in this project.")
        if asset.checksum_sha256 != node.get("source_checksum"):
            raise ValidationError("The approved source Asset changed; please request a new plan.")
        current = {
            "status": asset.technical_metadata_status,
            "width": asset.width, "height": asset.height,
            "has_audio_stream": asset.has_audio_stream,
        }
        fingerprint = render_compiler._canonical_hash(current)
        if fingerprint != node.get("source_metadata_fingerprint"):
            raise ValidationError(
                "The approved source metadata changed; please request a new plan."
            )


async def _execute_standard_step(
    session: AsyncSession,
    project_id: uuid.UUID,
    step: PlanStep,
    outputs: dict[str, str | None],
    step_runs: list[dict],
    plan_row: AgentPlan,
    summary: dict,
) -> None:
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
    output_ids = [item["asset_id"] for item in (job.output or {}).get("outputs", [])]
    output_id = output_ids[0] if len(output_ids) == 1 else None
    outputs[step.id] = output_id
    run = {
        "step_id": step.id, "capability_id": step.capability_id,
        "job_id": str(job.id), "status": job.status.value,
        "output_asset_id": output_id,
        "output_asset_ids": output_ids,
    }
    step_runs.append(run)
    _update_summary_step(
        summary, step.id, status=job.status.value, job_id=str(job.id),
        output_asset_id=output_id, output_asset_ids=output_ids,
    )
    plan_row.execution_summary = copy.deepcopy(summary)
    flag_modified(plan_row, "execution_summary")
    plan_row.step_runs = list(step_runs)
    await session.commit()
    if job.status == JobStatus.failed:
        raise RuntimeError(f"Step '{step.id}' failed: {job.error}")


async def _execute_legacy_steps(
    session: AsyncSession,
    plan: Plan,
    project_id: uuid.UUID,
    outputs: dict[str, str | None],
    step_runs: list[dict],
) -> None:
    """Preserve old persisted-plan behavior when no optimizer summary exists."""
    for step in plan.steps:
        asset_id, job_input = _build_job_input(step, outputs)
        job = await jobs_svc.create_job(
            session,
            JobCreate(capability_id=step.capability_id, project_id=project_id,
                      asset_id=asset_id, input=job_input),
        )
        job = await jobs_svc.execute_job(session, job.id)
        output_ids = [item["asset_id"] for item in (job.output or {}).get("outputs", [])]
        output_id = output_ids[0] if output_ids else None
        outputs[step.id] = output_id
        step_runs.append({
            "step_id": step.id, "capability_id": step.capability_id,
            "job_id": str(job.id), "status": job.status.value,
            "output_asset_id": output_id, "output_asset_ids": output_ids,
        })
        if job.status == JobStatus.failed:
            raise RuntimeError(f"Step '{step.id}' failed: {job.error}")


def _update_summary_step(summary: dict, step_id: str, **values: Any) -> None:
    for item in summary.get("steps", []):
        if item.get("step_id") == step_id:
            item.update(values)
            return


async def _execute_render_group(
    session: AsyncSession,
    plan_row: AgentPlan,
    project_id: uuid.UUID,
    group: dict,
) -> tuple[Job, str | None]:
    """Execute and persist one physical Agent-only render-group Job."""
    source_asset = await get_asset(session, uuid.UUID(group["source_asset_id"]))
    if source_asset.project_id != project_id:
        raise ValidationError("Render input is not in the Agent plan's project.")
    job_input = {
        "render_group_id": group["id"],
        "logical_step_ids": group["step_ids"],
        "operations": group["operations"],
        "identity_hash": group["identity_hash"],
        "optimizer_version": render_compiler.OPTIMIZER_VERSION,
    }
    job = Job(
        project_id=project_id, asset_id=source_asset.id,
        capability_id="agent.render_group", status=JobStatus.queued,
        input=job_input, progress=0,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    job.status = JobStatus.running
    job.started_at = datetime.now(UTC)
    await session.commit()

    work_dir = Path(tempfile.mkdtemp(prefix=f"uniche-agent-render-{job.id}-"))
    output_asset_id: str | None = None
    try:
        source_path = get_storage().get_path(source_asset.storage_path)
        output_name = f"agent-{plan_row.id}-{group['id']}.{group['output_extension']}"
        output_path = work_dir / output_name
        audio_present = await render_compiler.render_group(
            source_path, output_path, work_dir, group
        )
        ctx = JobContext(
            capability_id="agent.render_group", input_path=str(source_path),
            input_asset_meta={"media_type": source_asset.media_type.value},
            work_dir=work_dir, project_id=project_id, source_asset_id=source_asset.id,
        )
        result = HandlerResult(
            data={
                "operation": "agent.render_group",
                "render_group_id": group["id"],
                "logical_step_ids": group["step_ids"],
                "identity_hash": group["identity_hash"],
                "video_encode_count": 1,
                "audio_encode_count": int(audio_present and (
                    any(op["operation"]["kind"] == "temporaltrim" and not op["identity"]
                        for op in group["operations"])
                    or group["output_extension"] != group["source_extension"]
                )),
            },
            outputs=[OutputFile(output_path, output_name, MediaType.video)],
        )
        job.output = await jobs_svc._persist_outputs(session, job, ctx, result)
        output_asset_id = job.output["outputs"][0]["asset_id"]
        job.status = JobStatus.succeeded
        job.progress = 100
        job.finished_at = datetime.now(UTC)
        await session.commit()
    except Exception as exc:  # noqa: BLE001 - persist physical group failure
        logger.exception("Agent render group %s failed", group["id"])
        job.status = JobStatus.failed
        job.error = str(exc)
        job.finished_at = datetime.now(UTC)
        await session.commit()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    await session.refresh(job)
    return job, output_asset_id
