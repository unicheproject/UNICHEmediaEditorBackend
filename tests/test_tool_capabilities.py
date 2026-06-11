"""Tests for deterministic local-tool capabilities.

Registration/validation tests need no binaries. Execution tests are skipped
unless ffmpeg/convert are installed (they run for real in the Docker image).
"""

import io
import shutil
import subprocess

import pytest
from httpx import AsyncClient

API = "/api/v1"

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_CONVERT = shutil.which("convert") is not None

TOOL_IDS = {
    "video.trim", "video.split", "video.concat", "video.transcode", "video.mute",
    "video.crop", "video.resize", "video.thumbnail",
    "image.resize", "image.crop", "image.format", "image.colour.adjust",
    "audio.trim", "audio.concat", "audio.gain", "audio.normalize", "audio.fade",
    "audio.transcode",
}


# --- registration / metadata (no binaries) ---------------------------------


async def test_all_tool_capabilities_registered(client: AsyncClient) -> None:
    caps = (await client.get(f"{API}/capabilities")).json()
    by_id = {c["id"]: c for c in caps}
    assert set(by_id) >= TOOL_IDS
    for cid in TOOL_IDS:
        assert by_id[cid]["cost_class"] == "deterministic"
        assert by_id[cid]["enabled"] is True


async def test_supported_media_types(client: AsyncClient) -> None:
    caps = {c["id"]: c for c in (await client.get(f"{API}/capabilities")).json()}
    assert caps["video.thumbnail"]["supported_media_types"] == ["video"]
    assert caps["image.resize"]["supported_media_types"] == ["image"]
    assert caps["audio.trim"]["supported_media_types"] == ["audio"]


# --- validation (no binaries) ----------------------------------------------


async def test_media_type_mismatch_rejected(client: AsyncClient) -> None:
    pid = (await client.post(f"{API}/projects", json={"name": "V"})).json()["id"]
    files = {"file": ("photo.png", io.BytesIO(b"img"), "image/png")}
    aid = (
        await client.post(f"{API}/projects/{pid}/assets", files=files)
    ).json()["id"]
    # video.trim against an image asset -> 422
    resp = await client.post(
        f"{API}/jobs",
        json={"capability_id": "video.trim", "asset_id": aid, "input": {"start": 0, "end": 1}},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- execution (needs binaries) --------------------------------------------


def _run(args: list[str]) -> None:
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())


def _make_video(path: str) -> None:
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
        "-i", "testsrc=duration=2:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-shortest", path,
    ])


def _make_audio(path: str) -> None:
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
        "-i", "sine=frequency=440:duration=2", path,
    ])


def _make_image(path: str) -> None:
    _run(["convert", "-size", "64x48", "xc:red", path])


async def _upload(client: AsyncClient, pid: str, path: str, name: str, mime: str) -> str:
    with open(path, "rb") as fh:
        files = {"file": (name, fh.read(), mime)}
    resp = await client.post(f"{API}/projects/{pid}/assets", files=files)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def _run_job(client: AsyncClient, capability: str, asset_id: str, params: dict) -> dict:
    resp = await client.post(
        f"{API}/jobs",
        json={"capability_id": capability, "asset_id": asset_id, "input": params},
    )
    assert resp.status_code == 201, resp.text
    return (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_video_thumbnail_execution(client: AsyncClient, tmp_path) -> None:
    src = tmp_path / "v.mp4"
    _make_video(str(src))
    pid = (await client.post(f"{API}/projects", json={"name": "V"})).json()["id"]
    aid = await _upload(client, pid, str(src), "v.mp4", "video/mp4")

    job = await _run_job(client, "video.thumbnail", aid, {"timestamp": 1})
    assert job["status"] == "succeeded", job.get("error")
    out = job["output"]["outputs"][0]
    assert out["media_type"] == "image"

    # Output is a real, downloadable derived asset with provenance.
    meta = (await client.get(f"{API}/assets/{out['asset_id']}")).json()
    assert meta["source_asset_id"] == aid
    dl = await client.get(f"{API}/assets/{out['asset_id']}/download")
    assert dl.status_code == 200 and len(dl.content) > 0


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick (convert) not installed")
async def test_image_resize_execution(client: AsyncClient, tmp_path) -> None:
    src = tmp_path / "i.png"
    _make_image(str(src))
    pid = (await client.post(f"{API}/projects", json={"name": "I"})).json()["id"]
    aid = await _upload(client, pid, str(src), "i.png", "image/png")

    job = await _run_job(client, "image.resize", aid, {"width": 32, "height": 24})
    assert job["status"] == "succeeded", job.get("error")
    out = job["output"]["outputs"][0]
    assert out["media_type"] == "image"
    meta = (await client.get(f"{API}/assets/{out['asset_id']}")).json()
    assert meta["source_asset_id"] == aid


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_audio_trim_execution(client: AsyncClient, tmp_path) -> None:
    src = tmp_path / "a.wav"
    _make_audio(str(src))
    pid = (await client.post(f"{API}/projects", json={"name": "A"})).json()["id"]
    aid = await _upload(client, pid, str(src), "a.wav", "audio/wav")

    job = await _run_job(client, "audio.trim", aid, {"start": 0, "end": 1})
    assert job["status"] == "succeeded", job.get("error")
    out = job["output"]["outputs"][0]
    assert out["media_type"] == "audio"
