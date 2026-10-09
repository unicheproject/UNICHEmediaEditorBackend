"""Asset endpoints: upload, list, metadata, rename, download, soft delete.

Every route is project-scoped: routes with ``project_id`` in the path guard via
``require_project``; asset-id routes resolve the owning project via
``require_asset_access``. Both lazy-provision + authorize through the catalogue.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_asset_access, require_project
from app.core.database import get_session
from app.core.errors import ValidationError
from app.models.asset import Asset
from app.models.project import Project
from app.schemas.asset import (
    AssetBulkDeleteRequest,
    AssetBulkDeleteResponse,
    AssetRead,
    AssetUpdate,
)
from app.schemas.waveform import WaveformResponse
from app.services import assets as svc
from app.services.thumbnails import get_or_create_thumbnail
from app.services.waveforms import read_waveform
from app.workers.queue import enqueue_asset_probe

router = APIRouter(tags=["assets"])


async def get_asset_probe_enqueuer() -> Callable[[uuid.UUID], Awaitable[None]]:
    """Dependency seam for safely testing asynchronous upload probing."""
    return enqueue_asset_probe


@router.post(
    "/projects/{project_id}/assets",
    response_model=AssetRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_asset(
    project_id: uuid.UUID,
    request: Request,
    file: UploadFile = File(...),
    derived_from_job_id: uuid.UUID | None = Form(
        default=None,
        description=(
            "Successful same-project Job allowing this media type; "
            "client-materialized/possibly edited, not content attestation."
        ),
    ),
    session: AsyncSession = Depends(get_session),
    _project: Project = Depends(require_project),
    enqueue_probe: Callable[[uuid.UUID], Awaitable[None]] = Depends(get_asset_probe_enqueuer),
) -> AssetRead:
    if not file.filename:
        raise ValidationError("An uploaded file with a filename is required")
    if set(await request.form()) & {
        "provenance_kind", "ai_generated", "ai_provenance", "source_job_id", "source_asset_id",
    }:
        raise ValidationError("Asset provenance is determined by the server")
    asset = await svc.create_asset(
        session,
        project_id,
        original_filename=file.filename,
        src=file.file,
        content_type=file.content_type,
        derived_from_job_id=derived_from_job_id,
    )
    try:
        await enqueue_probe(asset.id)
    except Exception as exc:  # noqa: BLE001 - upload success is independent of probing
        from app.core.logging import get_logger

        get_logger(__name__).warning(
            "Could not enqueue technical metadata probe for asset %s (%s)",
            asset.id,
            type(exc).__name__,
        )
    return AssetRead.model_validate(asset)


@router.get("/projects/{project_id}/assets", response_model=list[AssetRead])
async def list_assets(
    project_id: uuid.UUID,
    include_intermediate: bool = False,
    session: AsyncSession = Depends(get_session),
    _project: Project = Depends(require_project),
) -> list[AssetRead]:
    assets = await svc.list_assets(
        session, project_id, include_intermediate=include_intermediate
    )
    return [AssetRead.model_validate(a) for a in assets]


@router.post(
    "/projects/{project_id}/assets/bulk-delete",
    response_model=AssetBulkDeleteResponse,
    summary="Soft-delete a batch of project Assets",
    description=(
        "Accepts 1–100 unique Asset IDs. Returns ordered deleted/not_found outcomes; "
        "missing, already deleted and other-project IDs are indistinguishable. "
        "All matching Assets are soft-deleted in one transaction; no binaries, "
        "Job history or provenance records are removed."
    ),
)
async def bulk_delete_assets(
    project_id: uuid.UUID,
    data: AssetBulkDeleteRequest,
    session: AsyncSession = Depends(get_session),
    _project: Project = Depends(require_project),
) -> AssetBulkDeleteResponse:
    return await svc.bulk_soft_delete_assets(session, project_id, data)


@router.get("/assets/{asset_id}", response_model=AssetRead)
async def get_asset(asset: Asset = Depends(require_asset_access)) -> AssetRead:
    return AssetRead.model_validate(asset)


@router.get("/assets/{asset_id}/download")
async def download_asset(
    asset: Asset = Depends(require_asset_access),
) -> FileResponse:
    path = svc.ensure_asset_binary_available(asset)
    return FileResponse(
        path,
        media_type=asset.mime_type,
        filename=asset.original_filename,
    )


@router.get("/assets/{asset_id}/thumbnail")
async def get_asset_thumbnail(
    asset: Asset = Depends(require_asset_access),
) -> FileResponse:
    svc.ensure_asset_binary_available(asset)
    path = await get_or_create_thumbnail(asset)
    return FileResponse(path, media_type="image/jpeg")


@router.get(
    "/assets/{asset_id}/waveform", response_model=WaveformResponse,
    responses={202: {"model": WaveformResponse, "description": "Enrichment pending"}},
)
async def get_asset_waveform(
    response: Response, asset: Asset = Depends(require_asset_access),
) -> WaveformResponse:
    result = read_waveform(asset)
    if result.status == "pending":
        response.status_code = status.HTTP_202_ACCEPTED
        response.headers["Retry-After"] = "5"
    return result


@router.patch("/assets/{asset_id}", response_model=AssetRead)
async def update_asset(
    asset_id: uuid.UUID,
    data: AssetUpdate,
    _asset: Asset = Depends(require_asset_access),
    session: AsyncSession = Depends(get_session),
) -> AssetRead:
    asset = await svc.update_asset(session, asset_id, data)
    return AssetRead.model_validate(asset)


@router.delete("/assets/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_asset(
    asset: Asset = Depends(require_asset_access),
    session: AsyncSession = Depends(get_session),
) -> None:
    await svc.soft_delete_asset(session, asset.id)
