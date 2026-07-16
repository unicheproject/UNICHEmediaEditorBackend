"""arq worker definition. Run with: arq app.workers.worker.WorkerSettings"""

from __future__ import annotations

from app.core.logging import configure_logging
from app.workers.queue import redis_settings
from app.workers.tasks import run_job, run_plan


async def startup(ctx: dict) -> None:
    configure_logging()


class WorkerSettings:
    functions = [run_job, run_plan]
    redis_settings = redis_settings()
    on_startup = startup
    max_jobs = 10
    job_timeout = 1800
    # Lets Job.abort() (app.workers.queue.cancel_job) actually interrupt a job
    # that's already running, not just prevent one that hasn't started yet.
    allow_abort_jobs = True
