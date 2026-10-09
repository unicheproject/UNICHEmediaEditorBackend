"""Asset library ordering/escaping against an opt-in disposable local PostgreSQL DB."""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import select, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from alembic import command
from app.core.config import settings
from app.models.asset import Asset
from app.models.enums import MediaType
from app.models.project import Project
from app.schemas.asset import AssetSearchQuery
from app.services.assets import search_assets

URL = os.environ.get("ASSET_SEARCH_TEST_DATABASE_URL")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not URL, reason="requires disposable local PostgreSQL Asset library database")
def test_postgres_asset_search(monkeypatch: pytest.MonkeyPatch) -> None:
    assert URL is not None
    parsed = make_url(URL)
    assert parsed.host in {"localhost", "127.0.0.1"}
    assert parsed.database == "asset_library_test"
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
        engine = create_async_engine(URL)
        try:
            async with AsyncSession(engine, expire_on_commit=False) as session:
                project_id, foreign_id = uuid.uuid4(), uuid.uuid4()
                session.add_all([
                    Project(id=project_id, name="Library project"),
                    Project(id=foreign_id, name="Other project"),
                ])
                await session.flush()
                ids = [uuid.UUID(int=i + 1) for i in range(5)]
                names = ["Bravo.png", "alpha.mp3", "ALPHA.mp4", "delta.srt", "echo.bin"]
                epoch = datetime(2026, 1, 1, tzinfo=UTC)
                session.add_all([
                    Asset(
                        id=aid, project_id=project_id, filename="stored.bin",
                        original_filename=names[i], media_type=media_type,
                        mime_type="application/octet-stream", extension="bin",
                        size_bytes=(5 - i) * 10, storage_path=f"test/{aid}",
                        checksum_sha256="0" * 64, provenance_kind="original",
                        ai_provenance="none", created_at=epoch + timedelta(seconds=i),
                    )
                    for i, (aid, media_type) in enumerate(zip(ids, MediaType, strict=True))
                ])
                await session.commit()

                for sort_by, ascending in {
                    "created_at": ids, "name": [ids[i] for i in [1, 2, 0, 3, 4]],
                    "size": ids[::-1], "type": [ids[i] for i in [1, 0, 3, 4, 2]],
                }.items():
                    for order in ["asc", "desc"]:
                        expected = ascending if order == "asc" else ascending[::-1]
                        actual = []
                        for offset in [0, 2, 4]:
                            rows, total = await search_assets(session, project_id, AssetSearchQuery(
                                sort_by=sort_by, sort_order=order, limit=2, offset=offset,
                            ))
                            assert total == 5
                            actual.extend(row.id for row in rows)
                        assert actual == expected

                for name, q in [
                    ("100%_ready.png", "%"), ("100%_ready.png", "_"),
                    ("folder/name.png", "/"), ("path\\name.png", "\\"),
                    ("archive's.png", "'"), ("Αρχείο.png", "αρχείο"), ("🔎.png", "🔎"),
                ]:
                    await session.execute(update(Asset).where(Asset.id == ids[0]).values(
                        original_filename=name,
                    ))
                    rows, total = await search_assets(session, project_id, AssetSearchQuery(q=q))
                    assert total == 1 and [row.id for row in rows] == [ids[0]]

                rows, total = await search_assets(session, project_id, AssetSearchQuery(
                    q=" ALPHA ", media_type=MediaType.video,
                ))
                assert total == 1 and [row.id for row in rows] == [ids[2]]
                rows, total = await search_assets(session, project_id, AssetSearchQuery(
                    offset=2_147_483_647,
                ))
                assert not rows and total == 5

                await session.execute(update(Asset).values(
                    created_at=epoch, original_filename="same", size_bytes=1,
                    media_type=MediaType.image,
                ))
                for sort_by in ["created_at", "name", "size", "type"]:
                    for order in ["asc", "desc"]:
                        actual = []
                        for offset in [0, 2, 4]:
                            rows, total = await search_assets(session, project_id, AssetSearchQuery(
                                sort_by=sort_by, sort_order=order, limit=2, offset=offset,
                            ))
                            assert total == 5
                            actual.extend(row.id for row in rows)
                        assert actual == (ids if order == "asc" else ids[::-1])

                for aid, values in [
                    (ids[0], {"deleted_at": epoch}),
                    (ids[1], {"is_intermediate": True}),
                    (ids[2], {"project_id": foreign_id}),
                    (ids[3], {"deleted_at": epoch, "is_intermediate": True}),
                ]:
                    await session.execute(update(Asset).where(Asset.id == aid).values(**values))
                rows, total = await search_assets(session, project_id, AssetSearchQuery())
                assert total == 1 and [row.id for row in rows] == [ids[4]]
                rows, total = await search_assets(session, project_id, AssetSearchQuery(
                    include_intermediate=True,
                ))
                assert total == 2 and {row.id for row in rows} == {ids[1], ids[4]}
                assert len((await session.scalars(select(Asset))).all()) == 5
        finally:
            await engine.dispose()

    asyncio.run(require_empty())
    command.upgrade(config, "head")
    try:
        asyncio.run(verify())
    finally:
        command.downgrade(config, "base")
    asyncio.run(require_empty())
