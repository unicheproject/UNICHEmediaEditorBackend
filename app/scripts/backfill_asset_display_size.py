"""Restartable backfill of displayed dimensions for already-probed Assets.

Assets probed before migration 0016 have coded width/height but no display
size; a rotated phone video or an EXIF-oriented photo needs a fresh probe to
tell. Only display_width/display_height are written.

Run inside the backend API container with:
    python -m app.scripts.backfill_asset_display_size
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from sqlalchemy import select

from app.core.database import async_session_factory
from app.models.asset import Asset
from app.models.enums import MediaType, TechnicalMetadataStatus
from app.services.media_metadata import probe_media_safely
from app.services.storage import get_storage


async def refresh_display_size(asset_id: uuid.UUID) -> str:
    """Probe one Asset and persist only its display size; returns the outcome."""
    async with async_session_factory() as session:
        asset = await session.get(Asset, asset_id)
        if asset is None or asset.deleted_at is not None or asset.display_width is not None:
            return "skipped"
        path = get_storage().get_path(asset.storage_path)
        metadata = await probe_media_safely(path, asset.media_type, asset_id=asset.id)
        if (
            metadata.status != TechnicalMetadataStatus.available
            or metadata.display_width is None
            or metadata.display_height is None
        ):
            return "unavailable"
        asset.display_width = metadata.display_width
        asset.display_height = metadata.display_height
        await session.commit()
        return "updated"


async def backfill(*, project_id: uuid.UUID | None, batch_size: int) -> dict[str, int]:
    counts = {"updated": 0, "unavailable": 0, "skipped": 0}
    last_id: uuid.UUID | None = None
    while True:
        async with async_session_factory() as session:
            stmt = select(Asset.id).where(
                Asset.deleted_at.is_(None),
                Asset.technical_metadata_status == TechnicalMetadataStatus.available.value,
                Asset.media_type.in_([MediaType.video, MediaType.image]),
                Asset.display_width.is_(None),
            )
            if project_id is not None:
                stmt = stmt.where(Asset.project_id == project_id)
            if last_id is not None:
                stmt = stmt.where(Asset.id > last_id)
            ids = list(
                (await session.execute(stmt.order_by(Asset.id).limit(batch_size)))
                .scalars()
                .all()
            )
        if not ids:
            break
        for asset_id in ids:
            try:
                outcome = await refresh_display_size(asset_id)
            except Exception:  # noqa: BLE001 - continue after per-asset DB/probe errors
                outcome = "unavailable"
            counts[outcome] += 1
        last_id = ids[-1]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", type=uuid.UUID)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    counts = asyncio.run(backfill(project_id=args.project_id, batch_size=args.batch_size))
    print(
        "Asset display size backfill complete: "
        f"updated={counts['updated']} unavailable={counts['unavailable']} "
        f"skipped={counts['skipped']}"
    )


if __name__ == "__main__":
    main()
