"""Conversational agent endpoints: sessions, messages, plan approval."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import Clarification, Plan
from app.core.database import get_session
from app.models.agent import AgentSession
from app.schemas.agent import (
    AgentMessageResponse,
    AgentPlanRead,
    AgentSessionCreate,
    AgentSessionRead,
    MessageCreate,
    TranscriptMessage,
)
from app.services import agent as svc
from app.workers.queue import enqueue_plan

router = APIRouter(prefix="/agent", tags=["agent"])

# Plan execution is enqueued; tests override this to run the plan eagerly.
PlanEnqueuer = Callable[[uuid.UUID], Awaitable[None]]


def get_plan_enqueuer() -> PlanEnqueuer:
    return enqueue_plan


def _session_read(s: AgentSession) -> AgentSessionRead:
    return AgentSessionRead(
        id=s.id,
        project_id=s.project_id,
        asset_ids=[uuid.UUID(a) for a in s.asset_ids],
        transcript=[
            TranscriptMessage(role=m.get("role", ""), content=m.get("content"))
            for m in s.messages
        ],
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


@router.post("/sessions", response_model=AgentSessionRead, status_code=status.HTTP_201_CREATED)
async def create_session(
    data: AgentSessionCreate, session: AsyncSession = Depends(get_session)
) -> AgentSessionRead:
    row = await svc.create_session(session, data.project_id, data.asset_ids)
    return _session_read(row)


@router.get("/sessions/{session_id}", response_model=AgentSessionRead)
async def get_session_detail(
    session_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> AgentSessionRead:
    row = await svc.get_session(session, session_id)
    return _session_read(row)


@router.post("/sessions/{session_id}/messages", response_model=AgentMessageResponse)
async def post_message(
    session_id: uuid.UUID,
    data: MessageCreate,
    session: AsyncSession = Depends(get_session),
) -> AgentMessageResponse:
    result, plan_row = await svc.post_message(session, session_id, data.content)
    if isinstance(result, Clarification):
        return AgentMessageResponse(
            type="clarification", question=result.question, missing=result.missing
        )
    assert isinstance(result, Plan) and plan_row is not None
    return AgentMessageResponse(type="plan", plan=AgentPlanRead.model_validate(plan_row))


@router.get("/plans/{plan_id}", response_model=AgentPlanRead)
async def get_plan(
    plan_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> AgentPlanRead:
    plan = await svc.get_plan(session, plan_id)
    return AgentPlanRead.model_validate(plan)


@router.post("/plans/{plan_id}/approve", response_model=AgentPlanRead)
async def approve_plan(
    plan_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    enqueuer: PlanEnqueuer = Depends(get_plan_enqueuer),
) -> AgentPlanRead:
    plan = await svc.approve_plan(session, plan_id)
    await enqueuer(plan.id)
    return AgentPlanRead.model_validate(plan)
