"""Server-authoritative editor provenance, independent of cost and initiation."""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from jsonschema import Draft202012Validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.capabilities import registry
from app.capabilities.validation import iter_media_asset_references
from app.core.errors import NotFoundError, ProviderError, ValidationError
from app.models.asset import Asset
from app.models.enums import AiContentEffect, AiProvenance, JobStatus, MediaType
from app.models.job import Job
from app.providers.validation import require_success_object


def contributing_asset_ids(job: Job) -> list[uuid.UUID]:
    """Read primary and schema-declared references, never arbitrary ID suffixes."""
    capability = registry.get(job.capability_id)
    ids = [job.asset_id] if job.asset_id is not None else []
    ids.extend(uuid.UUID(ref.asset_id) for ref in iter_media_asset_references(
        capability, job.input or {},
    ))
    return list(dict.fromkeys(ids))


async def output_ai_provenance(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    capability_ids: Iterable[str],
    asset_ids: Iterable[uuid.UUID],
) -> AiProvenance:
    """Direct effect wins; otherwise inherit recorded AI ancestry/uncertainty."""
    effects = {registry.get(cap_id).ai_content_effect for cap_id in capability_ids}
    states = set()
    for asset_id in set(asset_ids):
        asset = await session.get(Asset, asset_id)
        if asset is None or asset.project_id != project_id or asset.deleted_at is not None:
            raise ValidationError("Provenance input is not available in this project")
        states.add(asset.ai_provenance)
    if AiContentEffect.generated in effects:
        return AiProvenance.generated
    if AiContentEffect.modified in effects:
        return AiProvenance.modified
    if states & {AiProvenance.generated, AiProvenance.modified, AiProvenance.inherited}:
        return AiProvenance.inherited
    if AiProvenance.unknown in states:
        return AiProvenance.unknown
    return AiProvenance.none


async def materialization_job(
    session: AsyncSession,
    *,
    authorized_project_id: uuid.UUID,
    job_id: uuid.UUID,
    media_type: MediaType,
) -> Job:
    """Same authorized project only; conceal foreign/unscoped Job existence.

    The upload route's require_project guard runs before this lookup. No second
    project's permission or Job content is exposed to the caller.
    """
    job = await session.get(Job, job_id)
    if job is None or job.project_id != authorized_project_id:
        raise NotFoundError("Originating Job not found")
    if job.status != JobStatus.succeeded:
        raise ValidationError("Originating Job must have succeeded")
    capability = registry.get(job.capability_id)
    if media_type not in capability.client_materialization_media_types:
        raise ValidationError("This Job does not permit materialization of this media type")
    if not isinstance(job.output, dict) or not Draft202012Validator(
        capability.output_schema,
    ).is_valid(job.output):
        raise ValidationError("Originating Job has no valid materializable result")
    try:
        require_success_object(job.output)
    except ProviderError:
        raise ValidationError("Originating Job has no valid materializable result") from None
    return job
