"""Conservative backfill and FK lifecycle on disposable LOCAL PostgreSQL only."""

from __future__ import annotations

import asyncio
import io
import os
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from alembic import command
from app.core.config import settings
from app.models.asset import Asset
from app.models.enums import MediaType
from app.models.job import Job
from app.models.project import Project
from app.services.assets import create_asset

ROOT = Path(__file__).resolve().parents[1]
URL = os.environ.get("PROVENANCE_TEST_DATABASE_URL")


@pytest.mark.skipif(not URL, reason="requires disposable local PostgreSQL provenance database")
@pytest.mark.parametrize("previous", [None, "0013_asset_waveforms"])
def test_postgres_provenance_migration(monkeypatch, previous):
    # Explicit test-only database name and loopback prevent use against deployments.
    assert URL is not None
    parsed = make_url(URL)
    assert parsed.host in {"localhost", "127.0.0.1"}
    assert parsed.database == "me09_provenance_test"
    monkeypatch.setattr(settings, "database_url", URL)
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    project_id, source_id, derived_id, job_id = (uuid.uuid4() for _ in range(4))

    async def sql(work):
        engine = create_async_engine(URL)
        try:
            async with engine.begin() as conn:
                await work(conn)
        finally:
            await engine.dispose()

    async def require_empty(conn):
        assert await conn.scalar(sa.text("SELECT to_regclass('public.assets')")) is None

    asyncio.run(sql(require_empty))
    if previous:
        command.upgrade(config, previous)

        async def seed(conn):
            await conn.execute(
                sa.text("INSERT INTO projects(id, name) VALUES (:id, 'Historical project')"),
                {"id": project_id},
            )
            for asset_id, anchor in [(source_id, None), (derived_id, source_id)]:
                await conn.execute(
                    sa.text(
                        "INSERT INTO assets(id, project_id, filename, original_filename, "
                        "media_type, "
                        "mime_type, extension, size_bytes, storage_path, checksum_sha256, "
                        "source_asset_id) "
                        "VALUES (:id, :project, 'media', 'media', 'video', 'video/mp4', 'mp4', 1, "
                        "'test/media', :checksum, :anchor)"
                    ),
                    {"id": asset_id, "project": project_id, "checksum": "0" * 64, "anchor": anchor},
                )

        asyncio.run(sql(seed))
    command.upgrade(config, "head")

    async def verify(conn):
        assert await conn.scalar(sa.text("SELECT version_num FROM alembic_version")) == (
            "0014_asset_provenance"
        )
        if previous:
            rows = (
                (
                    await conn.execute(
                        sa.text(
                            "SELECT id, provenance_kind, ai_provenance, source_job_id "
                            "FROM assets ORDER BY id"
                        )
                    )
                )
                .mappings()
                .all()
            )
            values = {
                row["id"]: (row["provenance_kind"], row["ai_provenance"], row["source_job_id"])
                for row in rows
            }
            assert values == {
                source_id: ("unknown", "unknown", None),
                derived_id: ("derived", "unknown", None),
            }
        else:
            await conn.execute(sa.insert(Project).values(id=project_id, name="New project"))
        # Legacy writes default to unknown, never accidentally Original/non-AI.
        new_id = uuid.uuid4()
        await conn.execute(
            sa.insert(Asset).values(
                id=new_id,
                project_id=project_id,
                filename="upload",
                original_filename="upload",
                media_type=MediaType.video,
                mime_type="video/mp4",
                extension="mp4",
                size_bytes=1,
                storage_path="test/new",
                checksum_sha256="0" * 64,
            )
        )
        assert (
            await conn.execute(
                sa.select(Asset.provenance_kind, Asset.ai_provenance).where(Asset.id == new_id)
            )
        ).one() == ("unknown", "unknown")
        await conn.execute(
            sa.insert(Job).values(
                id=job_id,
                project_id=project_id,
                capability_id="media.titlecard",
                input={"text": "Title"},
                status="succeeded",
                progress=100,
            )
        )
        await conn.execute(
            sa.update(Asset)
            .where(Asset.id == new_id)
            .values(
                provenance_kind="derived",
                ai_provenance="generated",
                source_job_id=job_id,
            )
        )
        await conn.execute(sa.delete(Job).where(Job.id == job_id))
        assert (
            await conn.execute(
                sa.select(
                    Asset.provenance_kind,
                    Asset.ai_provenance,
                    Asset.source_job_id,
                ).where(Asset.id == new_id)
            )
        ).one() == ("derived", "generated", None)
        async with AsyncSession(bind=conn, expire_on_commit=False) as session:
            uploaded = await create_asset(
                session, project_id, original_filename="manual.srt",
                src=io.BytesIO(b"1\n00:00:00,000 --> 00:00:01,000\nSubtitle\n"),
                content_type="application/x-subrip",
            )
            assert (uploaded.provenance_kind, uploaded.ai_provenance, uploaded.source_job_id) == (
                "original", "none", None,
            )

    asyncio.run(sql(verify))
    # Bounded rollback of only the newly created test schema also checks reversibility.
    command.downgrade(config, "base")
    asyncio.run(sql(require_empty))
