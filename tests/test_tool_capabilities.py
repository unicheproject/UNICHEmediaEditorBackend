"""Tests for deterministic local-tool capabilities.

Registration/validation tests need no binaries. Execution tests are skipped
unless ffmpeg/convert are installed (they run for real in the Docker image).
"""

import importlib.util
import io
import os
import shutil
import subprocess

import pytest
from httpx import AsyncClient

from app.tools import ffmpeg, realesrgan
from tests.conftest import DEFAULT_ORG_ID

API = "/api/v1"

_slug_counter = {"n": 0}


async def _project(client: AsyncClient, name: str) -> str:
    _slug_counter["n"] += 1
    resp = await client.post(
        f"{API}/projects",
        json={"name": name, "slug": f"tool-{_slug_counter['n']}", "org_id": DEFAULT_ORG_ID},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_CONVERT = shutil.which("convert") is not None
HAVE_RNNOISE_MODEL = os.path.exists(ffmpeg.DEFAULT_RNNOISE_MODEL)
HAVE_REALESRGAN = os.path.exists(realesrgan.REALESRGAN_BIN) and os.path.exists(
    os.path.join(realesrgan.REALESRGAN_MODELS_DIR, f"{realesrgan.DEFAULT_MODEL}.bin")
)
HAVE_SCENEDETECT = importlib.util.find_spec("scenedetect") is not None

TOOL_IDS = {
    "video.trim", "video.split", "video.concat", "video.transcode", "video.mute",
    "video.crop", "video.resize", "video.thumbnail", "video.shot.detect",
    "image.resize", "image.crop", "image.format", "image.colour.adjust", "image.upscale",
    "audio.trim", "audio.concat", "audio.gain", "audio.normalize", "audio.fade",
    "audio.denoise", "audio.transcode",
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
    pid = await _project(client, "V")
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


def _make_shot_video(path: str, tmp_path) -> None:
    """Two visually distinct 2s/25fps segments concatenated, for shot-boundary detection.

    Short/low-fps clips can fall under PySceneDetect's default min_scene_len (15
    frames), merging the cut away, so this uses a longer/higher-fps pair than
    _make_video's throwaway fixtures.
    """
    blue = tmp_path / "_shot_blue.mp4"
    red = tmp_path / "_shot_red.mp4"
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
        "-i", "color=c=blue:duration=2:size=320x240:rate=25", str(blue),
    ])
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
        "-i", "color=c=red:duration=2:size=320x240:rate=25", str(red),
    ])
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(blue), "-i", str(red),
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
        path,
    ])


def _make_noisy_audio(path: str) -> None:
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-f", "lavfi", "-i", "anoisesrc=d=2:c=white:a=0.3",
        "-filter_complex", "amix=inputs=2:duration=first",
        path,
    ])


def _rms_level_db(path: str) -> float:
    proc = subprocess.run(
        ["ffmpeg", "-i", path, "-af", "astats", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    lines = [ln for ln in proc.stderr.splitlines() if "RMS level dB" in ln]
    return float(lines[-1].split(":")[-1].strip())


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
    pid = await _project(client, "V")
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


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.skipif(not HAVE_SCENEDETECT, reason="scenedetect not installed")
async def test_video_shot_detect_execution(client: AsyncClient, tmp_path) -> None:
    src = tmp_path / "shots.mp4"
    _make_shot_video(str(src), tmp_path)
    pid = await _project(client, "S")
    aid = await _upload(client, pid, str(src), "shots.mp4", "video/mp4")

    job = await _run_job(client, "video.shot.detect", aid, {})
    assert job["status"] == "succeeded", job.get("error")
    assert "outputs" not in job["output"]  # JSON-only capability, no derived asset

    shots = job["output"]["shots"]
    assert len(shots) == 2
    assert shots[0]["start"] == 0.0
    assert shots[0]["end"] == pytest.approx(2.0, abs=0.05)
    assert shots[1]["end"] == pytest.approx(4.0, abs=0.05)


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick (convert) not installed")
async def test_image_resize_execution(client: AsyncClient, tmp_path) -> None:
    src = tmp_path / "i.png"
    _make_image(str(src))
    pid = await _project(client, "I")
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
    pid = await _project(client, "A")
    aid = await _upload(client, pid, str(src), "a.wav", "audio/wav")

    job = await _run_job(client, "audio.trim", aid, {"start": 0, "end": 1})
    assert job["status"] == "succeeded", job.get("error")
    out = job["output"]["outputs"][0]
    assert out["media_type"] == "audio"


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.skipif(not HAVE_RNNOISE_MODEL, reason="RNNoise model not vendored at expected path")
async def test_audio_denoise_execution(client: AsyncClient, tmp_path) -> None:
    src = tmp_path / "noisy.wav"
    _make_noisy_audio(str(src))
    pid = await _project(client, "A")
    aid = await _upload(client, pid, str(src), "noisy.wav", "audio/wav")

    job = await _run_job(client, "audio.denoise", aid, {})
    assert job["status"] == "succeeded", job.get("error")
    out = job["output"]["outputs"][0]
    assert out["media_type"] == "audio"

    dl = await client.get(f"{API}/assets/{out['asset_id']}/download")
    assert dl.status_code == 200 and len(dl.content) > 0
    denoised = tmp_path / "denoised.wav"
    denoised.write_bytes(dl.content)

    assert _rms_level_db(str(denoised)) < _rms_level_db(str(src))


@pytest.mark.skipif(
    not HAVE_REALESRGAN, reason="realesrgan-ncnn-vulkan binary/model not vendored at expected path"
)
async def test_image_upscale_execution(client: AsyncClient, tmp_path) -> None:
    src = tmp_path / "i.png"
    _make_image(str(src))
    pid = await _project(client, "I")
    aid = await _upload(client, pid, str(src), "i.png", "image/png")

    job = await _run_job(client, "image.upscale", aid, {"scale": 2})
    assert job["status"] == "succeeded", job.get("error")
    out = job["output"]["outputs"][0]
    assert out["media_type"] == "image"

    dl = await client.get(f"{API}/assets/{out['asset_id']}/download")
    assert dl.status_code == 200 and len(dl.content) > 0
