"""Agent project authorization and execution-attempt lifecycle regressions."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.agent.schemas import Plan
from app.api.v1.agent import get_plan_enqueuer
from app.core.database import async_session_factory
from app.main import app
from app.models.agent import AgentPlanAttempt
from app.models.enums import AgentPlanStatus, JobStatus
from app.models.job import Job
from app.services import agent as agent_svc
from app.services.catalogue_client import get_catalogue_client

API = "/api/v1"
SOURCE = "22222222-2222-2222-2222-222222222222"


def _test_plan() -> Plan:
    return Plan.model_validate({
        "type": "plan",
        "summary": "Trim source",
        "steps": [{
            "id": "trim",
            "capability_id": "video.trim",
            "asset": SOURCE,
            "params": {"start": 0, "end": 1},
        }],
    })


async def _create_plan(
    client: AsyncClient, make_project, monkeypatch
) -> tuple[str, str, str]:
    project_id = await make_project("Agent lifecycle")
    session_response = await client.post(
        f"{API}/agent/sessions", json={"project_id": project_id, "asset_ids": []}
    )
    assert session_response.status_code == 201, session_response.text
    session_id = session_response.json()["id"]

    async def propose(*_args, **_kwargs):
        return _test_plan(), []

    monkeypatch.setattr("app.services.agent.llm.propose", propose)
    response = await client.post(
        f"{API}/agent/sessions/{session_id}/messages", json={"content": "Trim it"}
    )
    assert response.status_code == 200, response.text
    plan_id = response.json()["plan"]["id"]
    return project_id, session_id, plan_id


@pytest.mark.asyncio
async def test_agent_resources_require_catalogue_project_access(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    _, session_id, plan_id = await _create_plan(client, make_project, monkeypatch)
    fake_catalogue = app.dependency_overrides[get_catalogue_client]()
    session_body = (await client.get(f"{API}/agent/sessions/{session_id}")).json()
    fake_catalogue.projects.pop(session_body["project_id"])

    propose_called = False
    enqueue_calls = 0

    async def forbidden_propose(*_args, **_kwargs):
        nonlocal propose_called
        propose_called = True
        raise AssertionError("Unauthorized Agent content reached the planner")

    async def forbidden_enqueue(*_args):
        nonlocal enqueue_calls
        enqueue_calls += 1

    monkeypatch.setattr("app.services.agent.llm.propose", forbidden_propose)
    app.dependency_overrides[get_plan_enqueuer] = lambda: forbidden_enqueue

    session_get = await client.get(f"{API}/agent/sessions/{session_id}")
    session_post = await client.post(
        f"{API}/agent/sessions/{session_id}/messages", json={"content": "secret"}
    )
    plan_get = await client.get(f"{API}/agent/plans/{plan_id}")
    plan_approve = await client.post(f"{API}/agent/plans/{plan_id}/approve")

    assert [r.status_code for r in (session_get, session_post, plan_get, plan_approve)] == [404] * 4
    assert session_get.json() == {
        "error": {"code": "not_found", "message": "Agent session not found"}
    }
    assert session_post.json() == session_get.json()
    assert plan_get.json() == {
        "error": {"code": "not_found", "message": "Agent plan not found"}
    }
    assert plan_approve.json() == plan_get.json()
    assert propose_called is False
    assert enqueue_calls == 0


@pytest.mark.asyncio
async def test_duplicate_approval_enqueues_only_once_and_attempt_is_exposed(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    _, _, plan_id = await _create_plan(client, make_project, monkeypatch)
    entered = asyncio.Event()
    release = asyncio.Event()
    enqueued: list[tuple[uuid.UUID, uuid.UUID]] = []

    async def blocked_enqueue(plan_uuid: uuid.UUID, attempt_uuid: uuid.UUID) -> None:
        enqueued.append((plan_uuid, attempt_uuid))
        entered.set()
        await release.wait()

    app.dependency_overrides[get_plan_enqueuer] = lambda: blocked_enqueue
    first = asyncio.create_task(client.post(f"{API}/agent/plans/{plan_id}/approve"))
    await entered.wait()
    duplicate = await client.post(f"{API}/agent/plans/{plan_id}/approve")
    release.set()
    first_response = await first

    assert first_response.status_code == 200, first_response.text
    assert duplicate.status_code == 422
    assert len(enqueued) == 1
    plan_response = await client.get(f"{API}/agent/plans/{plan_id}")
    body = plan_response.json()
    assert body["status"] == "approved"
    assert len(body["attempts"]) == 1
    assert body["attempts"][0]["id"] == str(enqueued[0][1])
    assert body["attempts"][0]["status"] == "approved"


@pytest.mark.asyncio
async def test_failed_plan_reapproval_creates_distinct_attempts(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    _, _, plan_id = await _create_plan(client, make_project, monkeypatch)

    async def eager_attempt(plan_uuid: uuid.UUID, attempt_uuid: uuid.UUID) -> None:
        from app.agent.executor import execute_plan

        async with async_session_factory() as session:
            await execute_plan(session, plan_uuid, attempt_uuid)

    app.dependency_overrides[get_plan_enqueuer] = lambda: eager_attempt
    for expected_count in (1, 2):
        response = await client.post(f"{API}/agent/plans/{plan_id}/approve")
        assert response.status_code == 200, response.text
        current = await client.get(f"{API}/agent/plans/{plan_id}")
        assert current.json()["status"] == "failed"
        attempts = current.json()["attempts"]
        assert len(attempts) == expected_count
        assert attempts[-1]["attempt_number"] == expected_count
        assert attempts[-1]["status"] == "failed"
    assert attempts[0]["id"] != attempts[1]["id"]


@pytest.mark.asyncio
async def test_enqueue_failure_restores_approvable_state(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    _, _, plan_id = await _create_plan(client, make_project, monkeypatch)
    async with async_session_factory() as session:
        plan, attempt, previous, previous_error = await agent_svc.approve_plan(
            session, uuid.UUID(plan_id)
        )
        assert previous == AgentPlanStatus.proposed
        await agent_svc.release_unqueued_attempt(
            session, plan.id, attempt.id, previous, previous_error,
            "queue unavailable",
        )
        stored = await agent_svc.get_plan(session, plan.id)
        recorded = await session.get(AgentPlanAttempt, attempt.id)
        assert stored.status == AgentPlanStatus.proposed
        assert stored.active_attempt_id is None
        assert recorded is not None and recorded.status == "failed"


@pytest.mark.asyncio
async def test_duplicate_worker_delivery_has_one_execution_owner(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    _, _, plan_id = await _create_plan(client, make_project, monkeypatch)
    async with async_session_factory() as session:
        plan, attempt, _, _ = await agent_svc.approve_plan(session, uuid.UUID(plan_id))

    entered = asyncio.Event()
    release = asyncio.Event()
    physical_calls = 0

    async def one_physical_node(
        session, project_id, step, outputs, step_runs, plan_row, summary,
        *, attempt, is_intermediate=False,
    ) -> None:
        nonlocal physical_calls
        physical_calls += 1
        job_id = uuid.uuid4()
        session.add(Job(
            id=job_id,
            project_id=project_id,
            capability_id=step.capability_id,
            status=JobStatus.succeeded,
            input={},
            progress=100,
            finished_at=datetime.now(UTC),
        ))
        await session.commit()
        entered.set()
        await release.wait()
        outputs[step.id] = None
        step_runs.append({
            "step_id": step.id, "capability_id": step.capability_id,
            "job_id": str(job_id), "status": "succeeded",
            "output_asset_id": None, "output_asset_ids": [],
        })
        from app.agent.executor import _save_running_attempt, _update_summary_step

        _update_summary_step(summary, step.id, status="succeeded", job_id=str(job_id))
        await _save_running_attempt(session, plan_row, attempt, step_runs, summary)

    monkeypatch.setattr("app.agent.executor._execute_standard_step", one_physical_node)

    from app.agent.executor import execute_plan

    async def run_once():
        async with async_session_factory() as session:
            return await execute_plan(session, plan.id, attempt.id)

    owner = asyncio.create_task(run_once())
    await entered.wait()
    duplicate = await run_once()
    assert duplicate.status == AgentPlanStatus.running
    release.set()
    final = await owner

    assert final.status == AgentPlanStatus.succeeded
    assert physical_calls == 1
    async with async_session_factory() as session:
        attempts = list(await session.scalars(
            select(AgentPlanAttempt).where(AgentPlanAttempt.plan_id == plan.id)
        ))
        assert len(attempts) == 1
        assert attempts[0].status == "succeeded"
        assert len(attempts[0].job_ids) == 1


@pytest.mark.asyncio
async def test_cooperative_plan_cancellation_is_terminal_and_reapprovable(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    _, _, plan_id = await _create_plan(client, make_project, monkeypatch)
    async with async_session_factory() as session:
        plan, attempt, _, _ = await agent_svc.approve_plan(session, uuid.UUID(plan_id))

    async def cancel_node(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr("app.agent.executor._execute_standard_step", cancel_node)
    from app.agent.executor import execute_plan

    with pytest.raises(asyncio.CancelledError):
        async with async_session_factory() as session:
            await execute_plan(session, plan.id, attempt.id)

    async with async_session_factory() as session:
        stored = await agent_svc.get_plan(session, plan.id)
        recorded = await session.get(AgentPlanAttempt, attempt.id)
        assert stored.status == AgentPlanStatus.failed
        assert stored.error == "Agent plan execution was cooperatively cancelled."
        assert stored.active_attempt_id is None
        assert recorded is not None and recorded.status == "cancelled"
        assert recorded.finished_at is not None
        # A stale delivery of the completed/cancelled token is a safe no-op.
        assert await agent_svc.claim_plan_attempt(session, plan.id, attempt.id) is None


@pytest.mark.asyncio
async def test_cooperative_job_cancellation_marks_physical_job_terminal(
    make_project, monkeypatch
) -> None:
    from app.services import jobs as jobs_svc

    project_id = uuid.UUID(await make_project("cancelled physical Job"))
    async with async_session_factory() as session:
        job = Job(
            project_id=project_id,
            capability_id="video.trim",
            status=JobStatus.queued,
            input={},
            progress=0,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

        async def cancel_context(*_args, **_kwargs):
            raise asyncio.CancelledError

        monkeypatch.setattr(jobs_svc, "_build_job_context", cancel_context)
        with pytest.raises(asyncio.CancelledError):
            await jobs_svc.execute_job(session, job.id)
        await session.refresh(job)
        assert job.status == JobStatus.cancelled
        assert job.finished_at is not None
