"""Agent service: sessions, conversational planning, plan approval."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import llm
from app.agent.schemas import Plan, PlannerResult
from app.core.errors import NotFoundError, ValidationError
from app.models.agent import AgentPlan, AgentSession
from app.models.enums import AgentPlanStatus
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


async def _assets_by_id(session: AsyncSession, agent_session: AgentSession) -> dict[str, str]:
    out: dict[str, str] = {}
    for aid in agent_session.asset_ids:
        asset = await get_asset(session, uuid.UUID(aid))
        out[aid] = asset.media_type.value
    return out


async def post_message(
    session: AsyncSession, session_id: uuid.UUID, content: str
) -> tuple[PlannerResult, AgentPlan | None]:
    agent_session = await get_session(session, session_id)
    assets_by_id = await _assets_by_id(session, agent_session)

    result, new_messages = await llm.propose(
        list(agent_session.messages), content, assets_by_id
    )
    agent_session.messages = list(agent_session.messages) + new_messages

    plan_row: AgentPlan | None = None
    if isinstance(result, Plan):
        plan_row = AgentPlan(
            session_id=agent_session.id,
            summary=result.summary,
            status=AgentPlanStatus.proposed,
            steps=[s.model_dump() for s in result.steps],
            step_runs=[],
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


async def approve_plan(session: AsyncSession, plan_id: uuid.UUID) -> AgentPlan:
    plan = await get_plan(session, plan_id)
    if plan.status not in (AgentPlanStatus.proposed, AgentPlanStatus.failed):
        raise ValidationError(f"Plan '{plan_id}' is not approvable (status={plan.status.value})")
    plan.status = AgentPlanStatus.approved
    await session.commit()
    await session.refresh(plan)
    return plan
