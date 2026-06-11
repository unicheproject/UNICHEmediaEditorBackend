"""arq worker definition. Run with: arq app.workers.worker.WorkerSettings"""

from __future__ import annotations

from app.core.logging import configure_logging
from app.workers.queue import redis_settings
from app.workers.tasks import run_job


async def startup(ctx: dict) -> None:
    configure_logging()


class WorkerSettings:
    functions = [run_job]
    redis_settings = redis_settings()
    on_startup = startup
    max_jobs = 10
    job_timeout = 600
