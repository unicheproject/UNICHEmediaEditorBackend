"""Retention and physical cleanup for Temp Processed Assets."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.agent import AgentPlan, AgentSession
from app.models.asset import Asset
from app.models.enums import AgentPlanStatus, JobStatus
from app.models.job import Job
from app.services.storage import get_storage

logger = get_logger(__name__)
_ACTIVE_JOBS = (JobStatus.queued, JobStatus.running)
_ACTIVE_PLANS = (AgentPlanStatus.approved, AgentPlanStatus.running)


def _uuid_values(value: Any) -> set[uuid.UUID]:
    """Collect UUID-shaped values from persisted job/plan JSON structures."""
    found: set[uuid.UUID] = set()
    if isinstance(value, (list, tuple)):
        for item in value:
            found.update(_uuid_values(item))
    elif isinstance(value, dict):
        for item in value.values():
            found.update(_uuid_values(item))
    elif isinstance(value, (str, uuid.UUID)):
        with suppress(ValueError, TypeError, AttributeError):
            found.add(uuid.UUID(str(value)))
    return found


async def _protected_asset_ids(
    session: AsyncSession, candidates: set[uuid.UUID]
) -> set[uuid.UUID]:
    if not candidates:
        return set()
    protected: set[uuid.UUID] = set()

    jobs = await session.scalars(
        select(Job).where(Job.status.in_(_ACTIVE_JOBS))
    )
    for job in jobs:
        refs = _uuid_values(job.input)
        if job.asset_id is not None:
            refs.add(job.asset_id)
        protected.update(refs & candidates)

    plans = list(await session.scalars(select(AgentPlan)))
    active_plans = [plan for plan in plans if plan.status in _ACTIVE_PLANS]
    for plan in plans:
        protected.update(_uuid_values(plan.result_asset_ids) & candidates)
    if active_plans:
        sessions = await session.scalars(
            select(AgentSession).where(
                AgentSession.id.in_({plan.session_id for plan in active_plans})
            )
        )
        for agent_session in sessions:
            protected.update(_uuid_values(agent_session.asset_ids) & candidates)
        for plan in active_plans:
            protected.update(
                _uuid_values(
                    [plan.steps, plan.step_runs, plan.execution_summary]
                )
                & candidates
            )
    return protected


async def cleanup_expired_temp_assets(
    session: AsyncSession, *, now: datetime | None = None, batch_size: int = 100
) -> int:
    """Delete expired temp binaries while retaining every audit/database row.

    The operation is safe to repeat. Legacy intermediate Assets have no expiry
    timestamp and are intentionally not selected for automatic deletion.
    """
    current_time = now or datetime.now(UTC)
    result = await session.scalars(
        select(Asset)
        .where(
            Asset.is_intermediate.is_(True),
            Asset.temp_expires_at.is_not(None),
            Asset.temp_expires_at <= current_time,
            Asset.physical_expired_at.is_(None),
        )
        .order_by(Asset.temp_expires_at)
        .limit(batch_size)
        .with_for_update(skip_locked=True)
    )
    candidates = list(result)
    protected = await _protected_asset_ids(
        session, {asset.id for asset in candidates}
    )

    deleted = 0
    storage = get_storage()
    for asset in candidates:
        if asset.id in protected:
            continue
        try:
            await asyncio.to_thread(
                storage.delete_asset, asset.project_id, asset.id
            )
        except Exception as exc:  # noqa: BLE001 - preserve state and retry later
            logger.warning(
                "Could not expire Temp Processed Asset %s (%s)",
                asset.id,
                type(exc).__name__,
            )
            continue

        # Storage deletion is complete (or the object was already absent).
        # Keep this separate from deleted_at, which means user deletion.
        asset.physical_expired_at = current_time
        deleted += 1
    try:
        # Keep candidate row locks until every deletion decision is committed,
        # so job submission (which also locks referenced Assets) cannot race
        # cleanup and enqueue a consumer while its source is being removed.
        await session.commit()
    except Exception:
        await session.rollback()
        logger.exception("Could not record Temp Processed Asset expiry batch")
        return 0
    return deleted
