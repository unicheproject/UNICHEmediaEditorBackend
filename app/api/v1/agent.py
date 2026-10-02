"""Conversational agent endpoints: sessions, messages, plan approval."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import Clarification, Plan
from app.core.database import get_session
from app.core.errors import NotFoundError
from app.core.security import Principal, get_current_principal
from app.models.agent import AgentSession
from app.schemas.agent import (
    AgentMessageResponse,
    AgentPlanAttemptRead,
    AgentPlanRead,
    AgentSessionCreate,
    AgentSessionRead,
    MessageCreate,
    TranscriptMessage,
)
from app.services import agent as svc
from app.services import projects as projects_svc
from app.services.catalogue_client import CatalogueClient, get_catalogue_client
from app.workers.queue import enqueue_plan

router = APIRouter(prefix="/agent", tags=["agent"])

# Plan execution is enqueued; tests override this to run the plan eagerly.
PlanEnqueuer = Callable[[uuid.UUID, uuid.UUID], Awaitable[None]]


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


async def _authorized_session(
    session: AsyncSession,
    session_id: uuid.UUID,
    principal: Principal,
    catalogue: CatalogueClient,
) -> AgentSession:
    """Resolve project scope, then apply the same Catalogue gate as project APIs."""
    try:
        row = await svc.get_session(session, session_id)
        await projects_svc.require_project_access(
            session, catalogue, principal, row.project_id
        )
        return row
    except NotFoundError as exc:
        # Keep inaccessible and unknown session IDs indistinguishable.
        raise NotFoundError("Agent session not found") from exc


async def _plan_read(session: AsyncSession, plan) -> AgentPlanRead:
    response = AgentPlanRead.model_validate(plan)
    attempts = await svc.list_plan_attempts(session, plan.id)
    return response.model_copy(
        update={
            "attempts": [AgentPlanAttemptRead.model_validate(item) for item in attempts]
        }
    )


@router.post("/sessions", response_model=AgentSessionRead, status_code=status.HTTP_201_CREATED)
async def create_session(
    data: AgentSessionCreate,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
) -> AgentSessionRead:
    if data.project_id is not None:
        await projects_svc.require_project_access(
            session, catalogue, principal, data.project_id
        )
    row = await svc.create_session(session, data.project_id, data.asset_ids)
    return _session_read(row)


@router.get("/sessions/{session_id}", response_model=AgentSessionRead)
async def get_session_detail(
    session_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
) -> AgentSessionRead:
    row = await _authorized_session(session, session_id, principal, catalogue)
    return _session_read(row)


@router.post("/sessions/{session_id}/messages", response_model=AgentMessageResponse)
async def post_message(
    session_id: uuid.UUID,
    data: MessageCreate,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
) -> AgentMessageResponse:
    await _authorized_session(session, session_id, principal, catalogue)
    result, plan_row = await svc.post_message(session, session_id, data.content)
    if isinstance(result, Clarification):
        return AgentMessageResponse(
            type="clarification", question=result.question, missing=result.missing
        )
    assert isinstance(result, Plan) and plan_row is not None
    return AgentMessageResponse(type="plan", plan=await _plan_read(session, plan_row))


@router.get("/plans/{plan_id}", response_model=AgentPlanRead)
async def get_plan(
    plan_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
) -> AgentPlanRead:
    try:
        plan = await svc.get_plan(session, plan_id)
    except NotFoundError as exc:
        raise NotFoundError("Agent plan not found") from exc
    try:
        await _authorized_session(session, plan.session_id, principal, catalogue)
    except NotFoundError as exc:
        raise NotFoundError("Agent plan not found") from exc
    return await _plan_read(session, plan)


@router.post("/plans/{plan_id}/approve", response_model=AgentPlanRead)
async def approve_plan(
    plan_id: uuid.UUID,
    principal: Principal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_session),
    catalogue: CatalogueClient = Depends(get_catalogue_client),
    enqueuer: PlanEnqueuer = Depends(get_plan_enqueuer),
) -> AgentPlanRead:
    plan = await svc.get_plan(session, plan_id)
    try:
        await _authorized_session(session, plan.session_id, principal, catalogue)
    except NotFoundError as exc:
        raise NotFoundError("Agent plan not found") from exc
    plan, attempt, previous_status, previous_error = await svc.approve_plan(
        session, plan_id
    )
    try:
        await enqueuer(plan.id, attempt.id)
    except Exception:
        await svc.release_unqueued_attempt(
            session, plan.id, attempt.id, previous_status, previous_error,
            "Execution could not be enqueued.",
        )
        raise
    return await _plan_read(session, plan)
