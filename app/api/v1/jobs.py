"""Job endpoints: create (enqueue), retrieve, list-by-project, SSE progress."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable

from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.database import async_session_factory, get_session
from app.models.enums import JobStatus
from app.schemas.common import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, Page
from app.schemas.job import JobCreate, JobRead
from app.services import jobs as svc
from app.workers.queue import enqueue_job

router = APIRouter(tags=["jobs"])

# Enqueue is injected so tests can override it (e.g. run the job eagerly)
# without standing up Redis/arq. Routes never touch provider logic.
Enqueuer = Callable[[uuid.UUID], Awaitable[None]]


def get_enqueuer() -> Enqueuer:
    return enqueue_job


_TERMINAL = {JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled}


@router.post("/jobs", response_model=JobRead, status_code=status.HTTP_201_CREATED)
async def create_job(
    data: JobCreate,
    session: AsyncSession = Depends(get_session),
    enqueuer: Enqueuer = Depends(get_enqueuer),
) -> JobRead:
    job = await svc.create_job(session, data)
    await enqueuer(job.id)
    return JobRead.model_validate(job)


@router.get("/jobs/{job_id}", response_model=JobRead)
async def get_job(
    job_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> JobRead:
    job = await svc.get_job(session, job_id)
    return JobRead.model_validate(job)


@router.get("/projects/{project_id}/jobs", response_model=Page[JobRead])
async def list_project_jobs(
    project_id: uuid.UUID,
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> Page[JobRead]:
    jobs, total = await svc.list_jobs_for_project(
        session, project_id, limit=limit, offset=offset
    )
    return Page[JobRead](
        items=[JobRead.model_validate(j) for j in jobs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: uuid.UUID, request: Request) -> EventSourceResponse:
    """Server-Sent Events stream of a job's status/progress until terminal."""

    async def event_stream() -> AsyncGenerator[dict, None]:
        last: tuple | None = None
        while True:
            if await request.is_disconnected():
                break
            async with async_session_factory() as session:
                job = await svc.get_job(session, job_id)
                snapshot = (job.status, job.progress)
                if snapshot != last:
                    last = snapshot
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {
                                "id": str(job.id),
                                "status": job.status.value,
                                "progress": job.progress,
                                "error": job.error,
                            }
                        ),
                    }
                if job.status in _TERMINAL:
                    break
            await asyncio.sleep(1.0)

    return EventSourceResponse(event_stream())
