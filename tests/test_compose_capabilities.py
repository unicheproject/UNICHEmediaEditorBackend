"""Execution tests for composition capabilities (skip-guarded on binaries)."""

import os
import shutil
import subprocess

import pytest
from httpx import AsyncClient

from app.tools.imagemagick import DEFAULT_FONT

API = "/api/v1"

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_CONVERT = shutil.which("convert") is not None
HAVE_FONT = os.path.exists(DEFAULT_FONT)
NEED_ALL = HAVE_FFMPEG and HAVE_CONVERT and HAVE_FONT

pytestmark = pytest.mark.skipif(not NEED_ALL, reason="needs ffmpeg + convert + font")


def _run(args: list[str]) -> None:
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())


def _make_video(path: str) -> None:
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
        "-i", "testsrc=duration=2:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-shortest", path,
    ])


def _make_image(path: str, color: str) -> None:
    _run(["convert", "-size", "320x240", f"xc:{color}", path])


def _make_audio(path: str) -> None:
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
          "-i", "sine=frequency=330:duration=3", path])


async def _project(client: AsyncClient) -> str:
    return (await client.post(f"{API}/projects", json={"name": "compose"})).json()["id"]


async def _upload(client: AsyncClient, pid: str, path: str, name: str, mime: str) -> str:
    with open(path, "rb") as fh:
        files = {"file": (name, fh.read(), mime)}
    r = await client.post(f"{API}/projects/{pid}/assets", files=files)
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _run_job(client: AsyncClient, body: dict) -> dict:
    r = await client.post(f"{API}/jobs", json=body)
    assert r.status_code == 201, r.text
    return (await client.get(f"{API}/jobs/{r.json()['id']}")).json()


def _first_output(job: dict) -> dict:
    assert job["status"] == "succeeded", job.get("error")
    return job["output"]["outputs"][0]


async def test_titlecard(client: AsyncClient) -> None:
    pid = await _project(client)
    job = await _run_job(client, {
        "capability_id": "media.titlecard", "project_id": pid,
        "input": {"text": "Museum Highlights", "duration": 2, "width": 320, "height": 240},
    })
    out = _first_output(job)
    assert out["media_type"] == "video"


async def test_slideshow_then_compose(client: AsyncClient, tmp_path) -> None:
    pid = await _project(client)
    imgs = []
    for i, c in enumerate(["red", "green", "blue"]):
        p = tmp_path / f"i{i}.png"
        _make_image(str(p), c)
        imgs.append(await _upload(client, pid, str(p), f"i{i}.png", "image/png"))

    slideshow = await _run_job(client, {
        "capability_id": "image.slideshow", "project_id": pid,
        "input": {"asset_ids": imgs, "seconds_per_image": 1, "width": 320, "height": 240},
    })
    slide_asset = _first_output(slideshow)["asset_id"]

    title = await _run_job(client, {
        "capability_id": "media.titlecard", "project_id": pid,
        "input": {"text": "Intro", "duration": 1, "width": 320, "height": 240},
    })
    title_asset = _first_output(title)["asset_id"]

    # Compose: title card then slideshow into one 320x240 video.
    composed = await _run_job(client, {
        "capability_id": "video.compose", "project_id": pid,
        "input": {"asset_ids": [title_asset, slide_asset], "width": 320, "height": 240},
    })
    out = _first_output(composed)
    assert out["media_type"] == "video"
    meta = (await client.get(f"{API}/assets/{out['asset_id']}")).json()
    assert meta["source_asset_id"] is not None


async def test_subtitle_embed_soft(client: AsyncClient, tmp_path) -> None:
    pid = await _project(client)
    vid_path = tmp_path / "v.mp4"
    _make_video(str(vid_path))
    vid = await _upload(client, pid, str(vid_path), "v.mp4", "video/mp4")

    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nHello heritage\n", encoding="utf-8")
    sub = await _upload(client, pid, str(srt), "s.srt", "application/x-subrip")
    # uploaded subtitle is classified as the subtitle media type
    assert (await client.get(f"{API}/assets/{sub}")).json()["media_type"] == "subtitle"

    job = await _run_job(client, {
        "capability_id": "video.subtitle.embed", "project_id": pid,
        "asset_id": vid, "input": {"subtitle_asset_id": sub, "mode": "soft"},
    })
    assert _first_output(job)["media_type"] == "video"


async def test_audio_mix_under_video(client: AsyncClient, tmp_path) -> None:
    pid = await _project(client)
    vid_path, mus_path = tmp_path / "v.mp4", tmp_path / "m.mp3"
    _make_video(str(vid_path))
    _make_audio(str(mus_path))
    vid = await _upload(client, pid, str(vid_path), "v.mp4", "video/mp4")
    mus = await _upload(client, pid, str(mus_path), "m.mp3", "audio/mpeg")

    job = await _run_job(client, {
        "capability_id": "audio.mix", "project_id": pid, "asset_id": vid,
        "input": {"music_asset_id": mus, "music_volume": 0.3, "mode": "mix"},
    })
    assert _first_output(job)["media_type"] == "video"
