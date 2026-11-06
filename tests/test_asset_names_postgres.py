"""Per-project unique Asset names under concurrency, on opt-in disposable PostgreSQL."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from alembic import command
from app.core.config import settings
from app.models.asset import Asset
from app.models.enums import MediaType
from app.models.project import Project
from app.services.assets import lock_project_filenames, unique_filenames

URL = os.environ.get("ASSET_NAMES_TEST_DATABASE_URL")
ROOT = Path(__file__).resolve().parents[1]
WRITERS = 12


@pytest.mark.skipif(not URL, reason="requires disposable local PostgreSQL Asset names database")
def test_concurrent_publications_get_distinct_names(monkeypatch: pytest.MonkeyPatch) -> None:
    assert URL is not None
    parsed = make_url(URL)
    assert parsed.host in {"localhost", "127.0.0.1"}
    assert parsed.database == "asset_names_test"
    monkeypatch.setattr(settings, "database_url", URL)
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))

    async def require_empty() -> None:
        engine = create_async_engine(URL)
        try:
            async with engine.connect() as conn:
                assert await conn.scalar(text("SELECT to_regclass('public.assets')")) is None
        finally:
            await engine.dispose()

    async def verify() -> None:
        engine = create_async_engine(URL, pool_size=WRITERS)
        project_id = uuid.uuid4()
        try:
            async with AsyncSession(engine) as session:
                session.add(Project(id=project_id, name="Names"))
                await session.commit()
            start = asyncio.Event()

            async def publish(index: int) -> None:
                async with AsyncSession(engine) as session:
                    await start.wait()
                    await lock_project_filenames(session, project_id)
                    (name,) = await unique_filenames(session, project_id, ["Scan.JPG"])
                    # Widen the race window: the lock must hold until commit.
                    await asyncio.sleep(0.01)
                    asset_id = uuid.uuid4()
                    session.add(Asset(
                        id=asset_id, project_id=project_id, filename="scan.jpg",
                        original_filename=name, media_type=MediaType.image,
                        mime_type="image/jpeg", extension="jpg", size_bytes=1,
                        storage_path=f"test/{asset_id}", checksum_sha256="0" * 64,
                        provenance_kind="original", ai_provenance="none",
                    ))
                    await session.commit()

            tasks = [asyncio.create_task(publish(i)) for i in range(WRITERS)]
            start.set()
            await asyncio.gather(*tasks)
            async with AsyncSession(engine) as session:
                names = list(await session.scalars(
                    select(Asset.original_filename).where(Asset.project_id == project_id)
                ))
                assert sorted(names, key=str.lower) == sorted(
                    ["Scan.JPG", *[f"Scan_{n}.JPG" for n in range(2, WRITERS + 1)]],
                    key=str.lower,
                )
                assert await unique_filenames(session, project_id, ["scan.jpg", "SCAN.jpg"]) == [
                    f"scan_{WRITERS + 1}.jpg", f"SCAN_{WRITERS + 2}.jpg",
                ]
                # PostgreSQL lower() also folds non-ASCII names, like Python's.
                session.add(Asset(
                    id=uuid.uuid4(), project_id=project_id, filename="a.png",
                    original_filename="Αρχείο.png", media_type=MediaType.image,
                    mime_type="image/png", extension="png", size_bytes=1,
                    storage_path="test/greek", checksum_sha256="0" * 64,
                    provenance_kind="original", ai_provenance="none",
                ))
                await session.commit()
                assert await unique_filenames(session, project_id, ["αρχείο.png"]) == [
                    "αρχείο_2.png",
                ]
        finally:
            await engine.dispose()

    asyncio.run(require_empty())
    command.upgrade(config, "head")
    try:
        asyncio.run(verify())
    finally:
        command.downgrade(config, "base")
    asyncio.run(require_empty())
