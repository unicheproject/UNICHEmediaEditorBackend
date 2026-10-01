"""Tests for deterministic local-tool capabilities.

Registration/validation tests need no binaries. Execution tests are skipped
unless ffmpeg/convert are installed (they run for real in the Docker image).
"""

import importlib.util
import io
import json
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
# No CPU fallback by design (workspace CLAUDE.md) -- the execution test only
# runs where a real CUDA GPU is present.
_HAVE_CUDA = False
if importlib.util.find_spec("torch") is not None:
    import torch as _torch

    _HAVE_CUDA = _torch.cuda.is_available()
HAVE_REALESRGAN = os.path.exists(realesrgan.MODEL_PATH) and _HAVE_CUDA
HAVE_SCENEDETECT = importlib.util.find_spec("scenedetect") is not None

TOOL_IDS = {
    "video.trim", "video.split", "video.concat", "video.transcode", "video.mute",
    "video.crop", "video.resize", "video.thumbnail", "video.shot.detect",
    "image.resize", "image.crop", "image.format", "image.colour.adjust", "image.upscale",
    "audio.trim", "audio.concat", "audio.gain", "audio.normalize", "audio.fade",
    "audio.denoise", "audio.transcode",
    "video.audio.extract",
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
    for capability in ("audio.normalize", "audio.gain", "audio.fade", "audio.denoise"):
        assert caps[capability]["supported_media_types"] == ["video", "audio"]
    assert caps["video.audio.extract"]["supported_media_types"] == ["video"]


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

    # Video soundtrack operations accept audio/video, but never unrelated media.
    resp = await client.post(
        f"{API}/jobs",
        json={"capability_id": "audio.normalize", "asset_id": aid, "input": {}},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    resp = await client.post(
        f"{API}/jobs",
        json={"capability_id": "video.audio.extract", "asset_id": aid, "input": {}},
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


def _make_av_mp4(path: str) -> None:
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", path,
    ])


def _make_silent_mp4(path: str) -> None:
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
        "-i", "testsrc=duration=2:size=320x240:rate=15",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ])


def _ffprobe(path: str) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


def _make_color_video(path: str, color: str) -> None:
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
        "-i", f"color=c={color}:duration=1:size=320x240:rate=15", path,
    ])


def _frame_rgb(path: str, timestamp: float) -> tuple[int, int, int]:
    proc = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-ss", str(timestamp), "-i", path,
            "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True,
        check=True,
    )
    assert len(proc.stdout) == 3
    return proc.stdout[0], proc.stdout[1], proc.stdout[2]


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
async def test_video_concat_preserves_explicit_asset_order(
    client: AsyncClient, tmp_path
) -> None:
    pid = await _project(client, "Ordered video concat")
    asset_ids = []
    for label, color in [("a", "red"), ("b", "lime"), ("c", "blue")]:
        path = tmp_path / f"{label}.mp4"
        _make_color_video(str(path), color)
        asset_ids.append(await _upload(client, pid, str(path), path.name, "video/mp4"))

    response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": "video.concat",
            "project_id": pid,
            "input": {"asset_ids": asset_ids},
        },
    )
    assert response.status_code == 201, response.text
    job = (await client.get(f"{API}/jobs/{response.json()['id']}")).json()
    assert job["status"] == "succeeded", job.get("error")

    output = job["output"]["outputs"][0]
    download = await client.get(output["download_path"])
    assert download.status_code == 200, download.text
    rendered = tmp_path / "ordered.mp4"
    rendered.write_bytes(download.content)

    red, green, blue = (_frame_rgb(str(rendered), t) for t in (0.5, 1.5, 2.5))
    assert red[0] > red[1] + 50 and red[0] > red[2] + 50
    assert green[1] > green[0] + 50 and green[1] > green[2] + 50
    assert blue[2] > blue[0] + 50 and blue[2] > blue[1] + 50


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


@pytest.mark.parametrize(
    ("capability", "params"),
    [
        ("audio.normalize", {}),
        ("audio.gain", {"gain_db": 1.0}),
        ("audio.fade", {"fade_in": 0.1, "fade_out": 0.1}),
        ("audio.denoise", {}),
    ],
)
@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_audio_transforms_still_return_audio(
    client: AsyncClient, tmp_path, capability: str, params: dict
) -> None:
    if capability == "audio.denoise" and not HAVE_RNNOISE_MODEL:
        pytest.skip("RNNoise model not vendored at expected path")
    src = tmp_path / "input.wav"
    _make_audio(str(src))
    pid = await _project(client, "Audio transform")
    aid = await _upload(client, pid, str(src), src.name, "audio/wav")

    job = await _run_job(client, capability, aid, params)
    assert job["status"] == "succeeded", job.get("error")
    output = job["output"]["outputs"][0]
    assert output["media_type"] == "audio"
    meta = (await client.get(f"{API}/assets/{output['asset_id']}")).json()
    assert meta["media_type"] == "audio"
    assert meta["technical_metadata_status"] == "available"
    assert meta["has_audio_stream"] is True
    assert meta["has_video_stream"] is False
    downloaded = await client.get(output["download_path"])
    assert downloaded.status_code == 200 and downloaded.content
    rendered = tmp_path / f"{capability.rsplit('.', 1)[-1]}.wav"
    rendered.write_bytes(downloaded.content)
    assert any(s["codec_type"] == "audio" for s in _ffprobe(str(rendered))["streams"])


@pytest.mark.parametrize(
    ("capability", "params"),
    [
        ("audio.normalize", {}),
        ("audio.gain", {"gain_db": 1.0}),
        ("audio.fade", {"fade_in": 0.1, "fade_out": 0.1}),
        ("audio.denoise", {}),
    ],
)
@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_audio_transforms_preserve_video_stream(
    client: AsyncClient, tmp_path, capability: str, params: dict
) -> None:
    if capability == "audio.denoise" and not HAVE_RNNOISE_MODEL:
        pytest.skip("RNNoise model not vendored at expected path")
    src = tmp_path / "input.mp4"
    _make_av_mp4(str(src))
    source_probe = _ffprobe(str(src))
    source_video = next(s for s in source_probe["streams"] if s["codec_type"] == "video")
    pid = await _project(client, "Video soundtrack transform")
    aid = await _upload(client, pid, str(src), src.name, "video/mp4")

    job = await _run_job(client, capability, aid, params)
    assert job["status"] == "succeeded", job.get("error")
    output = job["output"]["outputs"][0]
    assert output["media_type"] == "video"
    meta = (await client.get(f"{API}/assets/{output['asset_id']}")).json()
    assert meta["media_type"] == "video"
    assert meta["technical_metadata_status"] == "available"
    assert meta["width"] == 320 and meta["height"] == 240
    downloaded = await client.get(output["download_path"])
    assert downloaded.status_code == 200 and downloaded.content
    rendered = tmp_path / f"{capability.rsplit('.', 1)[-1]}.mp4"
    rendered.write_bytes(downloaded.content)
    probe = _ffprobe(str(rendered))
    streams = probe["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    assert {s["codec_type"] for s in streams} == {"video", "audio"}
    assert video["codec_name"] == source_video["codec_name"]
    assert (video["width"], video["height"]) == (320, 240)
    assert float(probe["format"]["duration"]) == pytest.approx(
        float(source_probe["format"]["duration"]), abs=0.15
    )


@pytest.mark.parametrize(
    ("capability", "params"),
    [
        ("audio.normalize", {}),
        ("audio.gain", {"gain_db": 1.0}),
        ("audio.fade", {"fade_in": 0.1}),
        ("audio.denoise", {}),
    ],
)
@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_audio_transforms_fail_for_silent_video(
    client: AsyncClient, tmp_path, capability: str, params: dict
) -> None:
    src = tmp_path / "silent.mp4"
    _make_silent_mp4(str(src))
    pid = await _project(client, "Silent video transform")
    aid = await _upload(client, pid, str(src), src.name, "video/mp4")
    job = await _run_job(client, capability, aid, params)
    assert job["status"] == "failed"
    assert "no audio stream" in job["error"]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_video_audio_extract_returns_audio_asset(client: AsyncClient, tmp_path) -> None:
    src = tmp_path / "interview.mp4"
    _make_av_mp4(str(src))
    pid = await _project(client, "Audio extraction")
    aid = await _upload(client, pid, str(src), src.name, "video/mp4")

    job = await _run_job(client, "video.audio.extract", aid, {})
    assert job["status"] == "succeeded", job.get("error")
    output = job["output"]["outputs"][0]
    assert output["media_type"] == "audio"
    assert output["filename"].endswith(".m4a")
    meta = (await client.get(f"{API}/assets/{output['asset_id']}")).json()
    assert meta["media_type"] == "audio"
    assert meta["technical_metadata_status"] == "available"
    downloaded = await client.get(output["download_path"])
    rendered = tmp_path / "extracted.m4a"
    rendered.write_bytes(downloaded.content)
    probe = _ffprobe(str(rendered))
    assert {s["codec_type"] for s in probe["streams"]} == {"audio"}
    assert next(s for s in probe["streams"] if s["codec_type"] == "audio")["codec_name"] == "aac"


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_video_audio_extract_rejects_silent_video(client: AsyncClient, tmp_path) -> None:
    src = tmp_path / "silent.mp4"
    _make_silent_mp4(str(src))
    pid = await _project(client, "Silent audio extraction")
    aid = await _upload(client, pid, str(src), src.name, "video/mp4")
    job = await _run_job(client, "video.audio.extract", aid, {})
    assert job["status"] == "failed"
    assert "no audio stream" in job["error"]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_video_audio_ffmpeg_commands_map_streams_and_copy_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    async def has_audio(_src: str) -> bool:
        return True

    async def capture(args: list[str], **_kwargs) -> str:
        captured.append(args)
        return ""

    monkeypatch.setattr(ffmpeg, "_has_audio", has_audio)
    monkeypatch.setattr(ffmpeg.runner, "run", capture)
    await ffmpeg.video_audio_gain("source.mp4", "result.mp4", 2.0)
    args = captured[-1]
    assert args[args.index("-map") + 1] == "0:v:0"
    assert args[args.index("-map") + 3] == "0:a:0"
    assert args[args.index("-c:v") + 1] == "copy"
    assert args[args.index("-c:a") + 1] == "aac"
    assert args[args.index("-filter:a") + 1] == "volume=2.0dB"

    await ffmpeg.video_audio_extract("source.mp4", "result.m4a")
    args = captured[-1]
    assert args[args.index("-map") + 1] == "0:a:0"
    assert "-vn" in args and "-c:v" not in args
    assert args[args.index("-c:a") + 1] == "aac"


@pytest.mark.parametrize(
    ("extension", "codec"),
    [("mp4", "aac"), ("mov", "aac"), ("mkv", "aac"), ("webm", "libopus")],
)
def test_video_audio_codec_is_compatible_with_supported_container(
    extension: str, codec: str
) -> None:
    assert ffmpeg._video_audio_codec(f"output.{extension}")[0] == codec


@pytest.mark.skipif(
    not HAVE_REALESRGAN, reason="requires a CUDA GPU (no CPU fallback by design)"
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
