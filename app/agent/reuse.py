"""Conservative deterministic reuse for physical nodes of the same AgentPlan.

This is intentionally not a general cache: entries are Jobs produced by an
earlier failed/cancelled attempt of the same plan, in the same project.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.capabilities import registry
from app.capabilities.handlers.not_implemented import NotImplementedHandler
from app.core.errors import AppError, ValidationError
from app.models.agent import AgentPlanAttempt
from app.models.asset import Asset
from app.models.enums import JobStatus
from app.models.job import Job
from app.services import jobs as jobs_svc
from app.services.agent import list_plan_attempts
from app.services.assets import ensure_asset_binary_available, get_asset

# Explicit policy: only these local, single-Asset-output handlers have been
# reviewed for same-plan retry reuse. Changing output semantics requires a
# deliberate version bump. Providers, generators, multi-output, JSON-only,
# external-state and composition capabilities are deliberately absent.
REUSE_VERSIONS: dict[str, str] = {
    "video.trim": "1",
    "video.concat": "1",
    # Quality was historically accepted as an unknown extra parameter and
    # ignored. A retry with that same input now has different output semantics.
    "video.transcode": "2",
    "video.mute": "1",
    "video.crop": "1",
    "video.resize": "1",
    "video.thumbnail": "1",
    "image.resize": "1",
    "image.crop": "1",
    "image.format": "1",
    "image.colour.adjust": "1",
    "audio.trim": "1",
    "audio.concat": "1",
    "audio.gain": "1",
    "audio.normalize": "1",
    "audio.fade": "1",
    "audio.transcode": "1",
    "audio.denoise": "1",
    "video.audio.extract": "1",
    "audio.mix": "1",
    "video.lowerthird": "1",
    "video.subtitle.embed": "1",
}


@dataclass(frozen=True)
class ReusableOutput:
    job: Job
    attempt: AgentPlanAttempt
    asset: Asset


def _with_schema_defaults(value: Any, schema: dict[str, Any]) -> Any:
    """Apply declared defaults and normalize JSON values by their schema type."""
    kind = schema.get("type")
    if kind == "object" and isinstance(value, dict):
        result = dict(value)
        properties = schema.get("properties", {})
        for name, property_schema in properties.items():
            if name not in result and "default" in property_schema:
                result[name] = property_schema["default"]
            if name in result:
                result[name] = _with_schema_defaults(result[name], property_schema)
        return result
    if kind == "array" and isinstance(value, list):
        item_schema = schema.get("items", {})
        return [_with_schema_defaults(item, item_schema) for item in value]
    if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _asset_refs(asset_id: uuid.UUID | None, params: dict[str, Any]) -> list[uuid.UUID]:
    refs: list[uuid.UUID] = []
    if asset_id is not None:
        refs.append(asset_id)
    for raw in params.get("asset_ids", []) or []:
        refs.append(raw if isinstance(raw, uuid.UUID) else uuid.UUID(str(raw)))
    for key, value in params.items():
        if key.endswith("_asset_id") and value:
            refs.append(value if isinstance(value, uuid.UUID) else uuid.UUID(str(value)))
    return refs


async def build_execution_identity(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    capability_id: str,
    asset_id: uuid.UUID | None,
    params: dict[str, Any],
) -> str | None:
    """Build an identity after canonical validation and project-safe resolution.

    Missing content checksums or a handler outside the explicit policy are a
    conservative cache miss. Invalid job inputs still follow normal validation
    error behavior rather than being hidden by a cache lookup.
    """
    version = REUSE_VERSIONS.get(capability_id)
    if version is None:
        return None
    capability = registry.get(capability_id)
    if isinstance(registry.get_handler(capability_id), NotImplementedHandler):
        return None

    jobs_svc.validate_job_input(capability, params)
    resolved_project = await jobs_svc.resolve_job_project(
        session,
        project_id=project_id,
        asset_id=asset_id,
        job_input=params,
        lock_assets=True,
    )
    if resolved_project != project_id:
        raise ValidationError("Agent execution input is not in the plan's project.")
    await jobs_svc._validate_referenced_asset_media_types(session, capability, params)

    fingerprints: dict[str, dict[str, Any]] = {}
    for reference_id in dict.fromkeys(_asset_refs(asset_id, params)):
        asset = await get_asset(session, reference_id)
        if asset.project_id != project_id:
            raise ValidationError("Agent execution input is not in the plan's project.")
        if not asset.checksum_sha256:
            return None
        fingerprints[str(reference_id)] = {
            "sha256": asset.checksum_sha256,
            "media_type": asset.media_type.value,
            "extension": asset.extension,
            "mime_type": asset.mime_type,
            "size_bytes": asset.size_bytes,
        }

    properties = capability.input_schema.get("properties", {})
    canonical_params = _with_schema_defaults(params, capability.input_schema)
    if not isinstance(canonical_params, dict):
        return None
    for name, property_schema in properties.items():
        if name not in canonical_params:
            continue
        if not property_schema.get("x-media-asset") and not name.endswith("_asset_id"):
            continue
        value = canonical_params[name]
        if property_schema.get("type") == "array" and isinstance(value, list):
            canonical_params[name] = [
                {"content": fingerprints.get(str(item))}
                if str(item) in fingerprints else item
                for item in value
            ]
        elif value not in (None, ""):
            token = fingerprints.get(str(value))
            if token is None:
                return None
            canonical_params[name] = {"content": token}

    primary = fingerprints.get(str(asset_id)) if asset_id is not None else None
    try:
        return _canonical_json_hash({
            "kind": "agent-job-node",
            "project_id": str(project_id),
            "capability_id": capability_id,
            "capability_reuse_version": version,
            "primary_input": primary,
            "parameters": canonical_params,
            "output_policy_version": version,
        })
    except (TypeError, ValueError):
        # Unexpected/non-JSON parameters must not make reuse less conservative.
        return None


def render_group_identity(
    project_id: uuid.UUID, group: dict[str, Any], *, optimizer_version: str
) -> str | None:
    """Wrap the compiler's ordered group fingerprint in an execution identity."""
    if not group.get("identity_hash") or not group.get("source_checksum"):
        return None
    try:
        return _canonical_json_hash({
            "kind": "agent-render-group",
            "project_id": str(project_id),
            "optimizer_version": optimizer_version,
            "group_fingerprint": group["identity_hash"],
        })
    except (TypeError, ValueError):
        return None


async def find_reusable_output(
    session: AsyncSession,
    *,
    plan_id: uuid.UUID,
    current_attempt: AgentPlanAttempt,
    project_id: uuid.UUID,
    identity: str,
) -> ReusableOutput | None:
    """Find and lock a valid output from an earlier failed attempt of this plan."""
    earlier_attempts = [
        attempt
        for attempt in await list_plan_attempts(session, plan_id)
        if attempt.attempt_number < current_attempt.attempt_number
        and attempt.status in {"failed", "cancelled"}
    ]
    if not earlier_attempts:
        return None
    prior_ids = [attempt.id for attempt in earlier_attempts]
    rows = await session.scalars(
        select(Job)
        .where(
            Job.agent_plan_id == plan_id,
            Job.agent_attempt_id.in_(prior_ids),
            Job.execution_identity == identity,
            Job.project_id == project_id,
            Job.status == JobStatus.succeeded,
        )
        .order_by(Job.finished_at.desc())
    )
    attempts_by_id = {attempt.id: attempt for attempt in earlier_attempts}
    for job in rows:
        if job.agent_attempt_id is None:
            continue
        source_attempt = attempts_by_id.get(job.agent_attempt_id)
        if source_attempt is None or job.finished_at is None:
            continue
        output_items = (job.output or {}).get("outputs", [])
        if not isinstance(output_items, list) or len(output_items) != 1:
            continue
        item = output_items[0]
        try:
            output_id = uuid.UUID(str(item["asset_id"]))
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        result = await session.execute(
            select(Asset)
            .where(
                Asset.id == output_id,
                Asset.project_id == project_id,
                Asset.deleted_at.is_(None),
            )
            .with_for_update()
        )
        asset = result.scalar_one_or_none()
        if asset is None or asset.physical_expired_at is not None:
            continue
        expiry = asset.temp_expires_at
        if expiry is not None:
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=UTC)
            if expiry <= datetime.now(UTC):
                continue
        if not asset.checksum_sha256 or asset.size_bytes <= 0:
            continue
        if asset.technical_metadata_status == "pending":
            continue
        if str(item.get("media_type")) != asset.media_type.value:
            continue
        try:
            ensure_asset_binary_available(asset)
        except AppError:
            continue
        return ReusableOutput(job=job, attempt=source_attempt, asset=asset)
    return None
