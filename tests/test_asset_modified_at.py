"""Asset.modified_at: user-visible modification time and date-modified sorting."""

from __future__ import annotations

import io
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from alembic.config import Config
from httpx import AsyncClient
from sqlalchemy import update

from alembic import command
from app.core.config import settings
from app.core.database import async_session_factory
from app.models.asset import Asset
from app.services.media_metadata import probe_asset_metadata

API = "/api/v1"
ROOT = Path(__file__).resolve().parents[1]


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_migration_backfills_from_created_at(monkeypatch) -> None:
    output = io.StringIO()
    config = Config(output_buffer=output)
    config.set_main_option("script_location", str(ROOT / "alembic"))
    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://test:test@localhost/test")
    command.upgrade(config, "0014_asset_provenance:0015_asset_modified_at", sql=True)
    sql = output.getvalue()
    assert "ADD COLUMN modified_at TIMESTAMP WITH TIME ZONE" in sql
    assert "UPDATE assets SET modified_at = created_at" in sql
    assert "ALTER COLUMN modified_at SET NOT NULL" in sql
    assert "ALTER COLUMN modified_at SET DEFAULT now()" in sql
    assert sql.index("UPDATE assets") < sql.index("SET NOT NULL")


async def _upload(client: AsyncClient, pid: str, name: str) -> dict:
    resp = await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": (name, io.BytesIO(b"\x89PNG not really"), "image/png")},
    )
    assert resp.status_code == 201, resp.text
    body: dict = resp.json()
    return body


async def test_only_user_changes_advance_modified_at(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("modified-at")
    created = await _upload(client, pid, "photo.png")
    assert created["modified_at"] == created["created_at"]

    # Background analysis updates the row but is not a user modification.
    async with async_session_factory() as session:
        await probe_asset_metadata(session, uuid.UUID(created["id"]))
    probed = (await client.get(f"{API}/assets/{created['id']}")).json()
    assert probed["technical_metadata_status"] != "pending"
    assert probed["modified_at"] == created["modified_at"]

    same = await client.patch(
        f"{API}/assets/{created['id']}", json={"original_filename": "photo.png"}
    )
    assert same.json()["modified_at"] == created["modified_at"]

    renamed = await client.patch(
        f"{API}/assets/{created['id']}", json={"original_filename": "Portrait.png"}
    )
    assert renamed.status_code == 200
    assert _ts(renamed.json()["modified_at"]) > _ts(created["modified_at"])
    assert renamed.json()["created_at"] == created["created_at"]


async def test_search_sorts_by_modified_at(client: AsyncClient, make_project) -> None:
    pid = await make_project("modified-sort")
    first = await _upload(client, pid, "a.png")
    second = await _upload(client, pid, "b.png")
    third = await _upload(client, pid, "c.png")
    # SQLite timestamps have one-second resolution; pin distinct historical times
    # so ordering does not fall back to the random-UUID tie-breaker.
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    async with async_session_factory() as session:
        for offset, asset in enumerate((first, second, third)):
            moment = epoch + timedelta(minutes=offset)
            await session.execute(
                update(Asset).where(Asset.id == uuid.UUID(asset["id"]))
                .values(created_at=moment, modified_at=moment)
            )
        await session.commit()
    await client.patch(f"{API}/assets/{first['id']}", json={"original_filename": "a2.png"})

    async def order(sort_by: str, sort_order: str = "desc") -> list[str]:
        resp = await client.get(
            f"{API}/projects/{pid}/assets/search",
            params={"sort_by": sort_by, "sort_order": sort_order},
        )
        assert resp.status_code == 200, resp.text
        return [item["original_filename"] for item in resp.json()["items"]]

    assert (await order("modified_at"))[0] == "a2.png"
    assert (await order("created_at"))[-1] == "a2.png"
    assert (await order("modified_at", "asc"))[0] == second["original_filename"]
