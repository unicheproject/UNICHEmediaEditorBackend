"""Agent service: sessions, conversational planning, plan approval."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import llm
from app.agent.render import compile_execution_plan, reset_execution_summary
from app.agent.schemas import Plan, PlannerResult
from app.core.errors import NotFoundError, ValidationError
from app.models.agent import AgentPlan, AgentPlanAttempt, AgentSession
from app.models.asset import Asset
from app.models.enums import AgentPlanStatus, TechnicalMetadataStatus
from app.services.assets import get_asset
from app.services.projects import get_project


async def create_session(
    session: AsyncSession, project_id: uuid.UUID, asset_ids: list[uuid.UUID]
) -> AgentSession:
    await get_project(session, project_id)
    for aid in asset_ids:
        asset = await get_asset(session, aid)  # 404 if missing
        if asset.project_id != project_id:
            raise ValidationError(f"Asset '{aid}' is not in project '{project_id}'")
    row = AgentSession(
        project_id=project_id,
        asset_ids=[str(a) for a in asset_ids],
        messages=[],
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def get_session(session: AsyncSession, session_id: uuid.UUID) -> AgentSession:
    row = await session.get(AgentSession, session_id)
    if row is None:
        raise NotFoundError(f"Agent session '{session_id}' not found")
    return row


def _planning_asset_context(asset: Asset) -> dict:
    """Whitelisted, compact asset facts safe for planner context."""
    status = asset.technical_metadata_status
    available = status == TechnicalMetadataStatus.available.value
    fields = (
        "duration_seconds",
        "width",
        "height",
        "has_video_stream",
        "has_audio_stream",
        "video_codec",
        "audio_codec",
        "frame_rate",
        "audio_sample_rate",
        "audio_channels",
    )
    return {
        "media_type": asset.media_type.value,
        "technical_metadata_status": status,
        **{field: getattr(asset, field) if available else None for field in fields},
    }


async def _assets_for_planning(
    session: AsyncSession, agent_session: AgentSession
) -> tuple[dict[str, str], dict[str, dict], dict[str, dict]]:
    assets_by_id: dict[str, str] = {}
    asset_context: dict[str, dict] = {}
    asset_facts: dict[str, dict] = {}
    for aid in agent_session.asset_ids:
        asset = await get_asset(session, uuid.UUID(aid))
        assets_by_id[aid] = asset.media_type.value
        asset_context[aid] = _planning_asset_context(asset)
        asset_facts[aid] = {
            "media_type": asset.media_type.value,
            "extension": asset.extension,
            "checksum_sha256": asset.checksum_sha256,
            "technical_metadata_status": asset.technical_metadata_status,
            "width": asset.width,
            "height": asset.height,
            "has_audio_stream": asset.has_audio_stream,
        }
    return assets_by_id, asset_context, asset_facts


async def post_message(
    session: AsyncSession, session_id: uuid.UUID, content: str
) -> tuple[PlannerResult, AgentPlan | None]:
    agent_session = await get_session(session, session_id)
    assets_by_id, asset_context, asset_facts = await _assets_for_planning(
        session, agent_session
    )

    result, new_messages = await llm.propose(
        list(agent_session.messages), content, assets_by_id,
        asset_context=asset_context,
    )
    agent_session.messages = list(agent_session.messages) + new_messages

    plan_row: AgentPlan | None = None
    if isinstance(result, Plan):
        execution_summary = compile_execution_plan(result, asset_facts)
        plan_row = AgentPlan(
            session_id=agent_session.id,
            summary=result.summary,
            status=AgentPlanStatus.proposed,
            steps=[s.model_dump() for s in result.steps],
            step_runs=[],
            execution_summary=execution_summary,
            result_asset_ids=[],
        )
        session.add(plan_row)
    await session.commit()
    if plan_row is not None:
        await session.refresh(plan_row)
    return result, plan_row


async def latest_plan(session: AsyncSession, session_id: uuid.UUID) -> AgentPlan | None:
    result = await session.execute(
        select(AgentPlan)
        .where(AgentPlan.session_id == session_id)
        .order_by(AgentPlan.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_plan(session: AsyncSession, plan_id: uuid.UUID) -> AgentPlan:
    plan = await session.get(AgentPlan, plan_id)
    if plan is None:
        raise NotFoundError(f"Agent plan '{plan_id}' not found")
    return plan


async def approve_plan(
    session: AsyncSession, plan_id: uuid.UUID
) -> tuple[AgentPlan, AgentPlanAttempt, AgentPlanStatus, str | None]:
    plan = await get_plan(session, plan_id)
    if plan.status not in (AgentPlanStatus.proposed, AgentPlanStatus.failed):
        raise ValidationError(f"Plan '{plan_id}' is not approvable (status={plan.status.value})")

    previous_status = plan.status
    previous_error = plan.error
    attempt_number = int(
        await session.scalar(
            select(func.coalesce(func.max(AgentPlanAttempt.attempt_number), 0)).where(
                AgentPlanAttempt.plan_id == plan_id
            )
        )
        or 0
    ) + 1
    attempt_id = uuid.uuid4()
    claimed = await session.execute(
        update(AgentPlan)
        .where(AgentPlan.id == plan_id, AgentPlan.status == previous_status)
        .values(
            status=AgentPlanStatus.approved,
            active_attempt_id=attempt_id,
            error=None,
        )
    )
    if getattr(claimed, "rowcount", 0) != 1:
        await session.rollback()
        raise ValidationError(f"Plan '{plan_id}' is not approvable")

    attempt = AgentPlanAttempt(
        id=attempt_id,
        plan_id=plan_id,
        attempt_number=attempt_number,
        status="approved",
        execution_summary=reset_execution_summary(plan.execution_summary),
    )
    session.add(attempt)
    await session.commit()
    await session.refresh(plan)
    await session.refresh(attempt)
    return plan, attempt, previous_status, previous_error


async def release_unqueued_attempt(
    session: AsyncSession,
    plan_id: uuid.UUID,
    attempt_id: uuid.UUID,
    previous_status: AgentPlanStatus,
    previous_error: str | None,
    error: str,
) -> None:
    """Restore the pre-approval state only if the worker has not claimed it."""
    changed = await session.execute(
        update(AgentPlan)
        .where(
            AgentPlan.id == plan_id,
            AgentPlan.status == AgentPlanStatus.approved,
            AgentPlan.active_attempt_id == attempt_id,
        )
        .values(
            status=previous_status,
            active_attempt_id=None,
            error=previous_error,
        )
    )
    attempt = await session.get(AgentPlanAttempt, attempt_id)
    if getattr(changed, "rowcount", 0) == 1 and attempt is not None:
        attempt.status = "failed"
        attempt.error = error
        attempt.finished_at = datetime.now(UTC)
    await session.commit()


async def list_plan_attempts(
    session: AsyncSession, plan_id: uuid.UUID
) -> list[AgentPlanAttempt]:
    result = await session.scalars(
        select(AgentPlanAttempt)
        .where(AgentPlanAttempt.plan_id == plan_id)
        .order_by(AgentPlanAttempt.attempt_number)
    )
    return list(result)


async def claim_plan_attempt(
    session: AsyncSession, plan_id: uuid.UUID, attempt_id: uuid.UUID
) -> tuple[AgentPlan, AgentPlanAttempt] | None:
    """Atomically transfer an approved attempt to exactly one worker."""
    attempt = await session.get(AgentPlanAttempt, attempt_id)
    if attempt is None or attempt.plan_id != plan_id or attempt.status != "approved":
        await session.rollback()
        return None
    claimed = await session.execute(
        update(AgentPlan)
        .where(
            AgentPlan.id == plan_id,
            AgentPlan.status == AgentPlanStatus.approved,
            AgentPlan.active_attempt_id == attempt_id,
        )
        .values(status=AgentPlanStatus.running)
    )
    if getattr(claimed, "rowcount", 0) != 1:
        await session.rollback()
        return None
    attempt.status = "running"
    attempt.started_at = datetime.now(UTC)
    await session.commit()
    plan = await get_plan(session, plan_id)
    await session.refresh(attempt)
    return plan, attempt


async def claim_legacy_approved_plan(
    session: AsyncSession, plan_id: uuid.UUID
) -> tuple[AgentPlan, AgentPlanAttempt] | None:
    """Claim a pre-upgrade ARQ delivery that has no persisted attempt token.

    This path is safe only while the plan has no active token. New approvals
    always set one, so a stale legacy delivery cannot claim a later retry.
    """
    attempt_number = int(
        await session.scalar(
            select(func.coalesce(func.max(AgentPlanAttempt.attempt_number), 0)).where(
                AgentPlanAttempt.plan_id == plan_id
            )
        )
        or 0
    ) + 1
    plan = await session.get(AgentPlan, plan_id)
    if plan is None:
        await session.rollback()
        return None
    attempt_id = uuid.uuid4()
    started_at = datetime.now(UTC)
    claimed = await session.execute(
        update(AgentPlan)
        .where(
            AgentPlan.id == plan_id,
            AgentPlan.status == AgentPlanStatus.approved,
            AgentPlan.active_attempt_id.is_(None),
        )
        .values(status=AgentPlanStatus.running, active_attempt_id=attempt_id)
    )
    if getattr(claimed, "rowcount", 0) != 1:
        await session.rollback()
        return None
    attempt = AgentPlanAttempt(
        id=attempt_id,
        plan_id=plan_id,
        attempt_number=attempt_number,
        status="running",
        started_at=started_at,
        execution_summary=reset_execution_summary(plan.execution_summary),
    )
    session.add(attempt)
    await session.commit()
    plan = await get_plan(session, plan_id)
    await session.refresh(attempt)
    return plan, attempt


async def snapshot_plan_attempt(
    session: AsyncSession,
    attempt: AgentPlanAttempt,
    *,
    step_runs: list[dict],
    execution_summary: dict | None,
    result_asset_ids: list[str],
    status: str | None = None,
    error: str | None = None,
    finished_at: datetime | None = None,
) -> None:
    """Persist the latest truthful attempt view alongside the plan's live view."""
    attempt.step_runs = list(step_runs)
    attempt.execution_summary = execution_summary
    attempt.result_asset_ids = list(result_asset_ids)
    job_ids: list[str] = []
    for run in step_runs:
        job_id = run.get("job_id")
        if job_id and job_id not in job_ids:
            job_ids.append(job_id)
    if execution_summary:
        for node in execution_summary.get("nodes", []):
            job_id = node.get("job_id")
            if job_id and job_id not in job_ids:
                job_ids.append(job_id)
    attempt.job_ids = job_ids
    if status is not None:
        attempt.status = status
    if error is not None:
        attempt.error = error
    if finished_at is not None:
        attempt.finished_at = finished_at
    await session.flush()
