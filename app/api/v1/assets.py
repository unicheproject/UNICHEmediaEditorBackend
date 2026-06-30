"""Asset endpoints: upload, list, metadata, download, soft delete.

Every route is project-scoped: routes with ``project_id`` in the path guard via
``require_project``; asset-id routes resolve the owning project via
``require_asset_access``. Both lazy-provision + authorize through the catalogue.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_asset_access, require_project
from app.core.database import get_session
from app.core.errors import ValidationError
from app.models.asset import Asset
from app.models.project import Project
from app.schemas.asset import AssetRead
from app.services import assets as svc
from app.services.storage import get_storage

router = APIRouter(tags=["assets"])


@router.post(
    "/projects/{project_id}/assets",
    response_model=AssetRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_asset(
    project_id: uuid.UUID,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    _project: Project = Depends(require_project),
) -> AssetRead:
    if not file.filename:
        raise ValidationError("An uploaded file with a filename is required")
    asset = await svc.create_asset(
        session,
        project_id,
        original_filename=file.filename,
        src=file.file,
        content_type=file.content_type,
    )
    return AssetRead.model_validate(asset)


@router.get("/projects/{project_id}/assets", response_model=list[AssetRead])
async def list_assets(
    project_id: uuid.UUID,
    include_intermediate: bool = True,
    session: AsyncSession = Depends(get_session),
    _project: Project = Depends(require_project),
) -> list[AssetRead]:
    assets = await svc.list_assets(
        session, project_id, include_intermediate=include_intermediate
    )
    return [AssetRead.model_validate(a) for a in assets]


@router.get("/assets/{asset_id}", response_model=AssetRead)
async def get_asset(asset: Asset = Depends(require_asset_access)) -> AssetRead:
    return AssetRead.model_validate(asset)


@router.get("/assets/{asset_id}/download")
async def download_asset(
    asset: Asset = Depends(require_asset_access),
) -> FileResponse:
    path = get_storage().get_path(asset.storage_path)
    return FileResponse(
        path,
        media_type=asset.mime_type,
        filename=asset.original_filename,
    )


@router.delete("/assets/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_asset(
    asset: Asset = Depends(require_asset_access),
    session: AsyncSession = Depends(get_session),
) -> None:
    await svc.soft_delete_asset(session, asset.id)
