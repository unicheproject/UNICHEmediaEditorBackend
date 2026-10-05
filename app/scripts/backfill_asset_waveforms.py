"""Restartable waveform enrichment/repair without Redis or a startup scan."""

import argparse
import asyncio
import uuid

from sqlalchemy import select

from app.core.database import async_session_factory
from app.models.asset import Asset
from app.models.enums import MediaType
from app.services.waveforms import enrich_asset_waveform


async def backfill(*, project_id: uuid.UUID | None, batch_size: int) -> dict[str, int]:
    if not 1 <= batch_size <= 1000:
        raise ValueError("batch_size must be between 1 and 1000")
    counts = {"available": 0, "unavailable": 0, "skipped": 0}
    last_id: uuid.UUID | None = None
    while True:
        async with async_session_factory() as session:
            stmt = select(Asset.id).where(
                Asset.deleted_at.is_(None), Asset.physical_expired_at.is_(None),
                Asset.media_type.in_((MediaType.audio, MediaType.video)),
            )
            if project_id is not None:
                stmt = stmt.where(Asset.project_id == project_id)
            if last_id is not None:
                stmt = stmt.where(Asset.id > last_id)
            ids = list(await session.scalars(stmt.order_by(Asset.id).limit(batch_size)))
        if not ids:
            break
        for asset_id in ids:
            async with async_session_factory() as session:
                state = await enrich_asset_waveform(session, asset_id)
                counts[state.value if state is not None else "skipped"] += 1
        last_id = ids[-1]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", type=uuid.UUID)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 1000:
        parser.error("--batch-size must be between 1 and 1000")
    print(asyncio.run(backfill(project_id=args.project_id, batch_size=args.batch_size)))


if __name__ == "__main__":
    main()
