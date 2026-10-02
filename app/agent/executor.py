"""Execute an approved agent plan as a chained sequence of capability jobs.

Reuses services.jobs (create_job + execute_job) so every step is a real,
tracked Job with media-type validation, derived-asset creation and status — and
each step's output asset feeds dependent steps via '@stepN' references.
"""

from __future__ import annotations

import asyncio
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
from app.agent import reuse as reuse_svc
from app.agent.schemas import Plan, PlanStep, is_step_ref, step_ref_id
from app.capabilities.context import HandlerResult, JobContext, OutputFile
from app.core.errors import AppError, ValidationError
from app.core.logging import get_logger
from app.models.agent import AgentPlan, AgentPlanAttempt, AgentSession
from app.models.enums import AgentPlanStatus, JobStatus, MediaType
from app.models.job import Job
from app.schemas.job import JobCreate
from app.services import agent as agent_svc
from app.services import jobs as jobs_svc
from app.services.assets import ensure_asset_binary_available, get_asset

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


async def execute_plan(
    session: AsyncSession, plan_id: uuid.UUID, attempt_id: uuid.UUID | None
) -> AgentPlan:
    claimed = (
        await agent_svc.claim_plan_attempt(session, plan_id, attempt_id)
        if attempt_id is not None
        else await agent_svc.claim_legacy_approved_plan(session, plan_id)
    )
    if claimed is None:
        return await agent_svc.get_plan(session, plan_id)
    plan_row, attempt = claimed
    agent_session = await session.get(AgentSession, plan_row.session_id)
    if agent_session is None:
        plan_row.status = AgentPlanStatus.failed
        plan_row.error = "Agent session not found"
        await _finish_attempt(session, plan_row, attempt, "failed", plan_row.error)
        return plan_row
    project_id = agent_session.project_id
    plan = Plan.model_validate(
        {"type": "plan", "summary": plan_row.summary, "steps": plan_row.steps}
    )
    summary = render_compiler.reset_execution_summary(attempt.execution_summary)
    step_runs: list[dict] = []
    plan_row.step_runs = []
    plan_row.result_asset_ids = []
    plan_row.error = None
    if summary is not None:
        plan_row.execution_summary = copy.deepcopy(summary)
        attempt.execution_summary = copy.deepcopy(summary)
        flag_modified(plan_row, "execution_summary")
        await session.commit()

    try:
        if summary is not None:
            if summary.get("optimizer_version") != render_compiler.OPTIMIZER_VERSION:
                raise ValidationError(
                    "The approved render plan is stale; please request a new plan."
                )
            if summary.get("input_fingerprint") != render_compiler.plan_fingerprint(plan):
                raise ValidationError(
                    "The approved plan inputs changed; please request a new plan."
                )
            await _validate_render_sources(session, summary, project_id)

        consumed = _consumed_step_ids(plan)
        outputs: dict[str, str | None] = {}
        if summary is None:
            await _execute_legacy_steps(
                session, plan, project_id, outputs, step_runs,
                plan_row=plan_row, attempt=attempt,
            )
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
                    await _save_running_attempt(
                        session, plan_row, attempt, step_runs, summary
                    )
                    continue
                if node["kind"] == "render_group":
                    terminal_step = node["step_ids"][-1]
                    job, output_asset_id = await _execute_render_group(
                        session, plan_row, project_id, node,
                        is_intermediate=terminal_step in consumed,
                        step_runs=step_runs, summary=summary, attempt=attempt,
                    )
                    reused = bool(node.get("reused"))
                    node["job_id"] = None if reused else str(job.id)
                    if reused:
                        node["source_job_id"] = str(job.id)
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
                            "render_group_id": node["id"],
                            "job_id": None if reused else str(job.id),
                            "reused": reused and not is_identity,
                            "source_job_id": str(job.id) if reused and not is_identity else None,
                            "source_attempt_id": (
                                node.get("source_attempt_id")
                                if reused and not is_identity else None
                            ),
                            "output_asset_id": (
                                output_asset_id if step_id == terminal_active else None
                            ),
                        })
                        outputs[step_id] = output_asset_id
                        _update_summary_step(
                            summary, step_id, status=status, identity=is_identity,
                            job_id=None if reused else str(job.id),
                            reused=reused and not is_identity,
                            source_job_id=(
                                node.get("source_job_id") if reused and not is_identity else None
                            ),
                            source_attempt_id=(
                                node.get("source_attempt_id")
                                if reused and not is_identity else None
                            ),
                            output_asset_id=output_asset_id if step_id == terminal_active else None,
                        )
                    await _save_running_attempt(
                        session, plan_row, attempt, step_runs, summary
                    )
                    if job.status == JobStatus.cancelled:
                        raise asyncio.CancelledError
                    if job.status == JobStatus.failed:
                        raise RuntimeError(f"Render group '{node['id']}' failed: {job.error}")
                    continue

                step = steps_by_id[node["step_id"]]
                await _execute_standard_step(
                    session, project_id, step, outputs, step_runs, plan_row,
                    summary, attempt=attempt,
                    is_intermediate=step.id in consumed,
                )

        finals: list[str] = []
        for run in step_runs:
            if run.get("render_group_id"):
                continue
            out_assets = run.get("output_asset_ids") or ([run["output_asset_id"]]
                         if run.get("output_asset_id") else [])
            if run["step_id"] not in consumed:
                finals.extend(asset_id for asset_id in out_assets if asset_id not in finals)
        for node in (summary or {}).get("nodes", []):
            if node.get("kind") == "render_group" and node.get("output_asset_id"):
                terminal_step = node["step_ids"][-1]
                asset_id = node["output_asset_id"]
                if terminal_step not in consumed and asset_id not in finals:
                    finals.append(asset_id)
        for node in (summary or {}).get("nodes", []):
            if (node.get("kind") == "identity"
                    and any(step_id not in consumed for step_id in node["step_ids"])
                    and node["output_asset_id"] not in finals):
                finals.append(node["output_asset_id"])
        plan_row.status = AgentPlanStatus.succeeded
        plan_row.result_asset_ids = finals
        await _finish_attempt(session, plan_row, attempt, "succeeded", None, step_runs, summary)
    except asyncio.CancelledError:
        plan_row.status = AgentPlanStatus.failed
        plan_row.error = "Agent plan execution was cooperatively cancelled."
        await _finish_attempt(
            session, plan_row, attempt, "cancelled", plan_row.error, step_runs, summary
        )
        raise
    except Exception as exc:  # noqa: BLE001 - record failure on the plan
        logger.exception("Plan %s failed", plan_id)
        plan_row.status = AgentPlanStatus.failed
        plan_row.error = str(exc)
        await _finish_attempt(session, plan_row, attempt, "failed", str(exc), step_runs, summary)

    await session.refresh(plan_row)
    return plan_row


async def _save_running_attempt(
    session: AsyncSession, plan_row: AgentPlan, attempt: AgentPlanAttempt,
    step_runs: list[dict], summary: dict | None,
) -> None:
    plan_row.step_runs = list(step_runs)
    if summary is not None:
        plan_row.execution_summary = copy.deepcopy(summary)
        flag_modified(plan_row, "execution_summary")
    await agent_svc.snapshot_plan_attempt(
        session, attempt, step_runs=step_runs,
        execution_summary=summary if summary is not None else plan_row.execution_summary,
        result_asset_ids=[],
    )
    await session.commit()


async def _finish_attempt(
    session: AsyncSession, plan_row: AgentPlan, attempt: AgentPlanAttempt,
    status: str, error: str | None, step_runs: list[dict] | None = None,
    summary: dict | None = None,
) -> None:
    now = datetime.now(UTC)
    plan_row.active_attempt_id = None
    plan_row.step_runs = list(step_runs if step_runs is not None else plan_row.step_runs)
    if summary is not None:
        plan_row.execution_summary = copy.deepcopy(summary)
        flag_modified(plan_row, "execution_summary")
    await agent_svc.snapshot_plan_attempt(
        session, attempt, step_runs=plan_row.step_runs,
        execution_summary=plan_row.execution_summary,
        result_asset_ids=[str(value) for value in plan_row.result_asset_ids or []],
        status=status, error=error, finished_at=now,
    )
    await session.commit()


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
    *,
    attempt: AgentPlanAttempt,
    is_intermediate: bool = False,
) -> None:
    asset_id, job_input = _build_job_input(step, outputs)
    execution_identity = await reuse_svc.build_execution_identity(
        session,
        project_id=project_id,
        capability_id=step.capability_id,
        asset_id=asset_id,
        params=job_input,
    )
    reusable = None
    if execution_identity is not None:
        reusable = await reuse_svc.find_reusable_output(
            session,
            plan_id=plan_row.id,
            current_attempt=attempt,
            project_id=project_id,
            identity=execution_identity,
    )
    if reusable is not None:
        cached_output_id = str(reusable.asset.id)
        outputs[step.id] = cached_output_id
        step_runs.append({
            "step_id": step.id,
            "capability_id": step.capability_id,
            "job_id": None,
            "status": JobStatus.succeeded.value,
            "output_asset_id": cached_output_id,
            "output_asset_ids": [cached_output_id],
            "reused": True,
            "source_job_id": str(reusable.job.id),
            "source_attempt_id": str(reusable.attempt.id),
        })
        _update_summary_step(
            summary,
            step.id,
            status=JobStatus.succeeded.value,
            job_id=None,
            output_asset_id=cached_output_id,
            output_asset_ids=[cached_output_id],
            reused=True,
            source_job_id=str(reusable.job.id),
            source_attempt_id=str(reusable.attempt.id),
        )
        await _save_running_attempt(session, plan_row, attempt, step_runs, summary)
        logger.info(
            "Agent retry reuse hit plan=%s attempt=%s step=%s source_job=%s asset=%s",
            plan_row.id, attempt.id, step.id, reusable.job.id, reusable.asset.id,
        )
        return

    if execution_identity is not None:
        logger.debug(
            "Agent retry reuse miss plan=%s attempt=%s node=%s reason=no_valid_prior_output",
            plan_row.id, attempt.id, step.id,
        )
    job = await jobs_svc.create_job(
        session,
        JobCreate(
            capability_id=step.capability_id,
            project_id=project_id,
            asset_id=asset_id,
            input=job_input,
        ),
    )
    try:
        job = await jobs_svc.execute_job(
            session, job.id, is_intermediate_output=is_intermediate
        )
    except asyncio.CancelledError:
        await session.refresh(job)
        output_ids = [item["asset_id"] for item in (job.output or {}).get("outputs", [])]
        cancelled_run = {
            "step_id": step.id, "capability_id": step.capability_id,
            "job_id": str(job.id), "status": JobStatus.cancelled.value,
            "output_asset_id": None, "output_asset_ids": output_ids,
        }
        step_runs.append(cancelled_run)
        _update_summary_step(
            summary, step.id, status=JobStatus.cancelled.value,
            job_id=str(job.id), output_asset_id=None,
        )
        await _save_running_attempt(session, plan_row, attempt, step_runs, summary)
        raise
    output_ids = [item["asset_id"] for item in (job.output or {}).get("outputs", [])]
    output_id = output_ids[0] if len(output_ids) == 1 else None
    if execution_identity is not None and job.status == JobStatus.succeeded and output_id:
        try:
            persisted_asset = await get_asset(session, uuid.UUID(output_id))
            if (
                persisted_asset.project_id == project_id
                and persisted_asset.checksum_sha256
                and persisted_asset.size_bytes > 0
                and persisted_asset.physical_expired_at is None
                and persisted_asset.technical_metadata_status != "pending"
            ):
                ensure_asset_binary_available(persisted_asset)
                job.agent_plan_id = plan_row.id
                job.agent_attempt_id = attempt.id
                job.execution_identity = execution_identity
                await session.commit()
        except (AppError, ValueError):
            # Successful processing is still successful, but incomplete output
            # cannot be published as a future retry hit.
            await session.rollback()
            await session.refresh(job)
    outputs[step.id] = output_id
    run: dict[str, Any] = {
        "step_id": step.id, "capability_id": step.capability_id,
        "job_id": str(job.id), "status": job.status.value,
        "output_asset_id": output_id,
        "output_asset_ids": output_ids,
        "reused": False,
    }
    step_runs.append(run)
    _update_summary_step(
        summary, step.id, status=job.status.value, job_id=str(job.id),
        output_asset_id=output_id, output_asset_ids=output_ids,
        reused=False,
    )
    await _save_running_attempt(session, plan_row, attempt, step_runs, summary)
    if job.status == JobStatus.failed:
        raise RuntimeError(f"Step '{step.id}' failed: {job.error}")


async def _execute_legacy_steps(
    session: AsyncSession,
    plan: Plan,
    project_id: uuid.UUID,
    outputs: dict[str, str | None],
    step_runs: list[dict],
    *,
    plan_row: AgentPlan,
    attempt: AgentPlanAttempt,
) -> None:
    """Preserve old persisted-plan behavior when no optimizer summary exists."""
    consumed = _consumed_step_ids(plan)
    for step in plan.steps:
        asset_id, job_input = _build_job_input(step, outputs)
        job = await jobs_svc.create_job(
            session,
            JobCreate(capability_id=step.capability_id, project_id=project_id,
                      asset_id=asset_id, input=job_input),
        )
        try:
            job = await jobs_svc.execute_job(
                session, job.id, is_intermediate_output=step.id in consumed,
            )
        except asyncio.CancelledError:
            await session.refresh(job)
            step_runs.append({
                "step_id": step.id, "capability_id": step.capability_id,
                "job_id": str(job.id), "status": JobStatus.cancelled.value,
                "output_asset_id": None, "output_asset_ids": [],
            })
            await _save_running_attempt(session, plan_row, attempt, step_runs, None)
            raise
        output_ids = [item["asset_id"] for item in (job.output or {}).get("outputs", [])]
        output_id = output_ids[0] if output_ids else None
        outputs[step.id] = output_id
        step_runs.append({
            "step_id": step.id, "capability_id": step.capability_id,
            "job_id": str(job.id), "status": job.status.value,
            "output_asset_id": output_id, "output_asset_ids": output_ids,
        })
        await _save_running_attempt(session, plan_row, attempt, step_runs, None)
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
    *,
    is_intermediate: bool,
    step_runs: list[dict],
    summary: dict,
    attempt: AgentPlanAttempt,
) -> tuple[Job, str | None]:
    """Execute and persist one physical Agent-only render-group Job."""
    source_asset = await get_asset(session, uuid.UUID(group["source_asset_id"]))
    if source_asset.project_id != project_id:
        raise ValidationError("Render input is not in the Agent plan's project.")
    execution_identity = reuse_svc.render_group_identity(
        project_id, group, optimizer_version=render_compiler.OPTIMIZER_VERSION
    )
    if execution_identity is not None:
        reusable = await reuse_svc.find_reusable_output(
            session,
            plan_id=plan_row.id,
            current_attempt=attempt,
            project_id=project_id,
            identity=execution_identity,
        )
        if reusable is not None:
            group.update(
                job_id=None,
                status=JobStatus.succeeded.value,
                output_asset_id=str(reusable.asset.id),
                reused=True,
                source_job_id=str(reusable.job.id),
                source_attempt_id=str(reusable.attempt.id),
            )
            logger.info(
                "Agent retry reuse hit plan=%s attempt=%s group=%s source_job=%s asset=%s",
                plan_row.id, attempt.id, group["id"], reusable.job.id, reusable.asset.id,
            )
            return reusable.job, str(reusable.asset.id)
        logger.debug(
            "Agent retry reuse miss plan=%s attempt=%s node=%s reason=no_valid_prior_output",
            plan_row.id, attempt.id, group["id"],
        )
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
        source_path = ensure_asset_binary_available(source_asset)
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
        job.output = await jobs_svc._persist_outputs(
            session, job, ctx, result, is_intermediate=is_intermediate
        )
        output_asset_id = job.output["outputs"][0]["asset_id"]
        job.status = JobStatus.succeeded
        job.progress = 100
        job.finished_at = datetime.now(UTC)
        if execution_identity is not None:
            persisted_asset = await get_asset(session, uuid.UUID(output_asset_id))
            if (
                persisted_asset.project_id == project_id
                and persisted_asset.checksum_sha256
                and persisted_asset.size_bytes > 0
                and persisted_asset.physical_expired_at is None
                and persisted_asset.technical_metadata_status != "pending"
            ):
                ensure_asset_binary_available(persisted_asset)
                job.agent_plan_id = plan_row.id
                job.agent_attempt_id = attempt.id
                job.execution_identity = execution_identity
        await session.commit()
    except asyncio.CancelledError:
        await session.rollback()
        current_job = await session.get(Job, job.id)
        if current_job is not None and current_job.status != JobStatus.cancelled:
            current_job.status = JobStatus.cancelled
            current_job.error = "Render group was cooperatively cancelled."
            current_job.finished_at = datetime.now(UTC)
        group["job_id"] = str(job.id)
        group["status"] = JobStatus.cancelled.value
        group["output_asset_id"] = None
        for op in group["operations"]:
            step_id = op["step_id"]
            step_runs.append({
                "step_id": step_id,
                "capability_id": op["capability_id"],
                "status": "identity" if op["identity"] else JobStatus.cancelled.value,
                "identity": bool(op["identity"]),
                "render_group_id": group["id"],
                "job_id": str(job.id),
                "output_asset_id": None,
            })
        plan_row.step_runs = list(step_runs)
        plan_row.execution_summary = copy.deepcopy(summary)
        for node in plan_row.execution_summary.get("nodes", []):
            if node.get("id") == group["id"]:
                node.update(job_id=str(job.id), status=JobStatus.cancelled.value,
                            output_asset_id=None)
        flag_modified(plan_row, "execution_summary")
        await session.commit()
        raise
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
