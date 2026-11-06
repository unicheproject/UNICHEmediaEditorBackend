"""POST /assets/{id}/duplicate: a same-project copy (devfixes §5 asset library)."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import update

from app.api.v1.assets import get_asset_probe_enqueuer
from app.core.database import async_session_factory
from app.main import app
from app.models.asset import Asset
from app.models.enums import TechnicalMetadataStatus
from app.services.media_metadata import probe_asset_metadata

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_CONVERT = shutil.which("convert") is not None


async def _upload(client: AsyncClient, pid: str, path: Path, mime: str) -> dict:
    resp = await client.post(
        f"{API}/projects/{pid}/assets", files={"file": (path.name, path.read_bytes(), mime)}
    )
    assert resp.status_code == 201, resp.text
    async with async_session_factory() as session:
        await probe_asset_metadata(session, uuid.UUID(resp.json()["id"]))
    body: dict = (await client.get(f"{API}/assets/{resp.json()['id']}")).json()
    return body


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
async def test_duplicate_copies_media_metadata_and_lineage(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "plate.jpg"
    subprocess.run(["convert", "-size", "320x200", "xc:olive", str(src)], check=True)
    pid = await make_project("duplicate")
    source = await _upload(client, pid, src, "image/jpeg")

    resp = await client.post(f"{API}/assets/{source['id']}/duplicate")
    assert resp.status_code == 201, resp.text
    copy = resp.json()
    assert copy["id"] != source["id"]
    assert copy["original_filename"] == "plate_2.jpg"
    for field in ("project_id", "media_type", "size_bytes", "checksum_sha256", "width",
                  "height", "display_width", "provenance_kind", "ai_provenance",
                  "technical_metadata_status"):
        assert copy[field] == source[field], field
    assert copy["is_intermediate"] is False
    original = (await client.get(f"{API}/assets/{source['id']}/download")).content
    assert (await client.get(f"{API}/assets/{copy['id']}/download")).content == original
    assert (await client.get(f"{API}/assets/{copy['id']}/thumbnail")).status_code == 200

    again = (await client.post(f"{API}/assets/{source['id']}/duplicate")).json()
    assert again["original_filename"] == "plate_3.jpg"
    # Deleting the copy leaves the source untouched.
    assert (await client.delete(f"{API}/assets/{copy['id']}")).status_code == 204
    assert (await client.get(f"{API}/assets/{source['id']}/download")).content == original


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_duplicate_of_audio_gets_its_own_waveform(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=duration=2", str(src)], check=True)
    pid = await make_project("duplicate-audio")
    source = await _upload(client, pid, src, "audio/wav")
    copy = (await client.post(f"{API}/assets/{source['id']}/duplicate")).json()
    assert copy["duration_seconds"] == source["duration_seconds"]
    waveform = await client.get(f"{API}/assets/{copy['id']}/waveform")
    assert waveform.status_code == 200 and waveform.json()["status"] == "available"


async def test_duplicate_rejects_deleted_and_expired_assets(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("duplicate-missing")
    resp = await client.post(f"{API}/projects/{pid}/assets",
                             files={"file": ("a.srt", b"1\n", "application/x-subrip")})
    asset_id = resp.json()["id"]
    async with async_session_factory() as session:
        await session.execute(update(Asset).where(Asset.id == uuid.UUID(asset_id))
                              .values(physical_expired_at=datetime.now(UTC)))
        await session.commit()
    assert (await client.post(f"{API}/assets/{asset_id}/duplicate")).status_code == 410
    assert (await client.delete(f"{API}/assets/{asset_id}")).status_code == 204
    assert (await client.post(f"{API}/assets/{asset_id}/duplicate")).status_code == 404
    assert (await client.post(f"{API}/assets/{uuid.uuid4()}/duplicate")).status_code == 404


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
async def test_pending_duplicate_is_probed_after_it_is_saved(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "scan.png"
    subprocess.run(["convert", "-size", "20x12", "xc:green", str(src)], check=True)
    queued: list[tuple[uuid.UUID, str | None]] = []

    async def capture(asset_id: uuid.UUID) -> None:
        # The probe must find the committed row when it runs.
        async with async_session_factory() as session:
            row = await session.get(Asset, asset_id)
            queued.append((asset_id, row.technical_metadata_status if row else None))

    app.dependency_overrides[get_asset_probe_enqueuer] = lambda: capture
    pid = await make_project("duplicate-pending")
    source = (await client.post(
        f"{API}/projects/{pid}/assets", files={"file": (src.name, src.read_bytes(), "image/png")}
    )).json()
    assert source["technical_metadata_status"] == "pending"

    resp = await client.post(f"{API}/assets/{source['id']}/duplicate")
    assert resp.status_code == 201, resp.text
    copy = resp.json()
    assert copy["technical_metadata_status"] == "pending"
    assert queued == [
        (uuid.UUID(source["id"]), "pending"), (uuid.UUID(copy["id"]), "pending"),
    ]

    # The existing probe completes the copy on its own.
    async with async_session_factory() as session:
        status = await probe_asset_metadata(session, uuid.UUID(copy["id"]))
    assert status == TechnicalMetadataStatus.available
    probed = (await client.get(f"{API}/assets/{copy['id']}")).json()
    assert (probed["width"], probed["height"]) == (20, 12)
    assert probed["checksum_sha256"] == source["checksum_sha256"]
    assert probed["provenance_kind"] == source["provenance_kind"]


@pytest.mark.parametrize("status", ["available", "unavailable"])
async def test_settled_metadata_is_copied_without_a_new_probe(
    client: AsyncClient, make_project, status: str
) -> None:
    queued: list[uuid.UUID] = []

    async def capture(asset_id: uuid.UUID) -> None:
        queued.append(asset_id)

    pid = await make_project(f"duplicate-{status}")
    asset_id = (await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": ("notes.srt", b"1\n", "application/x-subrip")},
    )).json()["id"]
    async with async_session_factory() as session:
        await session.execute(update(Asset).where(Asset.id == uuid.UUID(asset_id))
                              .values(technical_metadata_status=status))
        await session.commit()

    app.dependency_overrides[get_asset_probe_enqueuer] = lambda: capture
    copy = (await client.post(f"{API}/assets/{asset_id}/duplicate")).json()
    assert copy["technical_metadata_status"] == status
    assert queued == []


async def test_duplicate_succeeds_when_its_probe_cannot_be_queued(
    client: AsyncClient, make_project
) -> None:
    async def unavailable_queue(_asset_id: uuid.UUID) -> None:
        raise ConnectionError("queue unavailable")

    pid = await make_project("duplicate-queue-down")
    source = (await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": ("notes.srt", b"1\n", "application/x-subrip")},
    )).json()
    app.dependency_overrides[get_asset_probe_enqueuer] = lambda: unavailable_queue
    resp = await client.post(f"{API}/assets/{source['id']}/duplicate")
    # As for uploads: the copy exists and stays pending for the metadata backfill.
    assert resp.status_code == 201, resp.text
    assert resp.json()["technical_metadata_status"] == "pending"
    assert resp.json()["checksum_sha256"] == source["checksum_sha256"]
