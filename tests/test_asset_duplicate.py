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

from app.core.database import async_session_factory
from app.models.asset import Asset
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
