"""Restartable backfill for technical metadata on existing Assets.

Run inside the backend API container with:
    python -m app.scripts.backfill_asset_metadata
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from sqlalchemy import select

from app.core.database import async_session_factory
from app.models.asset import Asset
from app.models.enums import TechnicalMetadataStatus
from app.services.media_metadata import probe_asset_metadata


async def backfill(*, project_id: uuid.UUID | None, batch_size: int) -> dict[str, int]:
    counts = {"available": 0, "unavailable": 0, "skipped": 0}
    last_id: uuid.UUID | None = None
    while True:
        async with async_session_factory() as session:
            stmt = select(Asset.id).where(
                Asset.deleted_at.is_(None),
                Asset.technical_metadata_status != TechnicalMetadataStatus.available.value,
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
            async with async_session_factory() as session:
                try:
                    status = await probe_asset_metadata(session, asset_id)
                except Exception:  # noqa: BLE001 - continue after per-asset DB/probe errors
                    counts["unavailable"] += 1
                    continue
            if status == TechnicalMetadataStatus.available:
                counts["available"] += 1
            elif status == TechnicalMetadataStatus.unavailable:
                counts["unavailable"] += 1
            else:
                counts["skipped"] += 1
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
        "Asset metadata backfill complete: "
        f"available={counts['available']} unavailable={counts['unavailable']} "
        f"skipped={counts['skipped']}"
    )


if __name__ == "__main__":
    main()
