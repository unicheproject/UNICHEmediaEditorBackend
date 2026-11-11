"""arq worker definition. Run with: arq app.workers.worker.WorkerSettings"""

from __future__ import annotations

from arq import cron

from app.core.logging import configure_logging
from app.workers.queue import WORKER_JOB_TIMEOUT_SECONDS, redis_settings
from app.workers.tasks import (
    cleanup_temp_assets,
    reconcile_abandoned_work,
    run_asset_probe,
    run_job,
    run_plan,
)

# Abandoned-work recovery every 5 minutes (arq cron runs are unique per time).
RECONCILE_MINUTES = set(range(0, 60, 5))


async def startup(ctx: dict) -> None:
    configure_logging()
    # Recover work stranded by a previous crash; never prevents startup.
    await reconcile_abandoned_work(ctx)


class WorkerSettings:
    functions = [run_job, run_plan, run_asset_probe]
    cron_jobs = [
        cron(cleanup_temp_assets, minute=0),
        cron(reconcile_abandoned_work, minute=RECONCILE_MINUTES),
    ]
    redis_settings = redis_settings()
    on_startup = startup
    max_jobs = 10
    job_timeout = WORKER_JOB_TIMEOUT_SECONDS
    # Lets Job.abort() (app.workers.queue.cancel_job) actually interrupt a job
    # that's already running, not just prevent one that hasn't started yet.
    allow_abort_jobs = True
