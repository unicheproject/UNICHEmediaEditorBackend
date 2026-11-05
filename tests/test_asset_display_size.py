"""Asset display_width/display_height: rotation- and orientation-aware sizes."""

from __future__ import annotations

import io
import shutil
import uuid
from pathlib import Path

import pytest
from alembic.config import Config
from httpx import AsyncClient
from sqlalchemy import update

from alembic import command
from app.core.config import settings
from app.core.database import async_session_factory
from app.models.asset import Asset
from app.models.enums import MediaType
from app.scripts import backfill_asset_display_size
from app.services.media_metadata import probe_asset_metadata, probe_media
from tests.test_image_crop import _image, _with_exif_orientation
from tests.test_video_crop_thumbnail import _rotate
from tests.test_video_trim import _make_video

API = "/api/v1"
ROOT = Path(__file__).resolve().parents[1]
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_CONVERT = shutil.which("convert") is not None


def test_migration_adds_nullable_display_columns(monkeypatch) -> None:
    output = io.StringIO()
    config = Config(output_buffer=output)
    config.set_main_option("script_location", str(ROOT / "alembic"))
    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://test:test@localhost/test")
    command.upgrade(config, "0015_asset_modified_at:0016_asset_display_size", sql=True)
    sql = output.getvalue()
    assert "ADD COLUMN display_width INTEGER" in sql
    assert "ADD COLUMN display_height INTEGER" in sql
    assert "NOT NULL" not in sql.split("ADD COLUMN display_width", 1)[1].split(";", 1)[0]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_video_probe_reports_rotated_display_size(tmp_path: Path) -> None:
    plain, rotated = tmp_path / "plain.mp4", tmp_path / "rotated.mp4"
    _make_video(plain)
    _rotate(plain, rotated)
    straight = await probe_media(plain, MediaType.video)
    turned = await probe_media(rotated, MediaType.video)
    assert (straight.width, straight.height) == (320, 240)
    assert (straight.display_width, straight.display_height) == (320, 240)
    assert (turned.width, turned.height) == (320, 240)  # coded size is unchanged
    assert (turned.display_width, turned.display_height) == (240, 320)


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
async def test_image_probe_applies_exif_orientation(tmp_path: Path) -> None:
    plain = tmp_path / "plain.jpg"
    _image(plain, "300x200", "blue")
    rotated = tmp_path / "rotated.jpg"
    rotated.write_bytes(_with_exif_orientation(plain.read_bytes(), 6))
    straight = await probe_media(plain, MediaType.image)
    turned = await probe_media(rotated, MediaType.image)
    assert (straight.display_width, straight.display_height) == (300, 200)
    assert (turned.width, turned.height) == (300, 200)
    assert (turned.display_width, turned.display_height) == (200, 300)


async def _upload_probed(client: AsyncClient, pid: str, path: Path, mime: str) -> dict:
    resp = await client.post(
        f"{API}/projects/{pid}/assets", files={"file": (path.name, path.read_bytes(), mime)}
    )
    assert resp.status_code == 201, resp.text
    asset_id = resp.json()["id"]
    async with async_session_factory() as session:
        await probe_asset_metadata(session, uuid.UUID(asset_id))
    body: dict = (await client.get(f"{API}/assets/{asset_id}")).json()
    return body


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_api_exposes_display_size_and_crop_admission_is_exact(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    plain, rotated = tmp_path / "plain.mp4", tmp_path / "rotated.mp4"
    _make_video(plain)
    _rotate(plain, rotated)
    pid = await make_project("display-size")
    asset = await _upload_probed(client, pid, rotated, "video/mp4")
    assert (asset["width"], asset["height"]) == (320, 240)
    assert (asset["display_width"], asset["display_height"]) == (240, 320)

    # Fits the coded orientation only: previously admitted, now rejected up front.
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "video.crop", "asset_id": asset["id"],
        "input": {"x": 0, "y": 0, "width": 320, "height": 240},
    })
    assert resp.status_code == 422, resp.text
    assert "(240×320)" in resp.json()["error"]["message"]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_assets_without_display_size_keep_tolerant_admission_and_backfill(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    plain, rotated = tmp_path / "plain.mp4", tmp_path / "rotated.mp4"
    _make_video(plain)
    _rotate(plain, rotated)
    pid = await make_project("display-size-legacy")
    asset = await _upload_probed(client, pid, rotated, "video/mp4")
    asset_id = uuid.UUID(asset["id"])
    async with async_session_factory() as session:  # as probed before migration 0016
        await session.execute(
            update(Asset).where(Asset.id == asset_id)
            .values(display_width=None, display_height=None)
        )
        await session.commit()

    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "video.crop", "asset_id": asset["id"],
        "input": {"x": 0, "y": 0, "width": 320, "height": 240},
    })
    assert resp.status_code == 201, resp.text  # tolerant; execution rejects it
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "failed" and "(240×320)" in job["error"]

    counts = await backfill_asset_display_size.backfill(project_id=uuid.UUID(pid), batch_size=1)
    assert counts == {"updated": 1, "unavailable": 0, "skipped": 0}
    refreshed = (await client.get(f"{API}/assets/{asset['id']}")).json()
    assert (refreshed["display_width"], refreshed["display_height"]) == (240, 320)
    assert refreshed["modified_at"] == asset["modified_at"]  # not a user change
    again = await backfill_asset_display_size.backfill(project_id=uuid.UUID(pid), batch_size=1)
    assert again == {"updated": 0, "unavailable": 0, "skipped": 0}
