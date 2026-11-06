"""Distinct, parameterised Asset names within a project (devfixes C-8)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.capabilities.handlers._util import num_tag, trim_tag
from app.services.assets import numbered_filename

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_CONVERT = shutil.which("convert") is not None


def test_name_helpers() -> None:
    assert numbered_filename("photo.jpg", 2) == "photo_2.jpg"
    assert numbered_filename("archive.tar.gz", 3) == "archive.tar_3.gz"
    assert numbered_filename("README", 2) == "README_2"
    assert [num_tag(v) for v in (10, 10.0, 10.5, 0.125, -3, 1 / 3)] == [
        "10", "10", "10.5", "0.125", "-3", "0.333",
    ]
    assert trim_tag(10, 240) == "trim_10-240"
    assert trim_tag(2.5, None) == "trim_2.5-end"


async def _upload(client: AsyncClient, pid: str, name: str, data: bytes, mime: str) -> dict:
    resp = await client.post(f"{API}/projects/{pid}/assets", files={"file": (name, data, mime)})
    assert resp.status_code == 201, resp.text
    body: dict = resp.json()
    return body


async def _job(client: AsyncClient, capability: str, asset_id: str, params: dict) -> list[str]:
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": capability, "asset_id": asset_id, "input": params,
    })
    assert resp.status_code == 201, resp.text
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded", job
    return [output["filename"] for output in job["output"]["outputs"]]


async def test_uploads_with_the_same_name_are_numbered(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("names-upload")
    first = await _upload(client, pid, "notes.srt", b"1\n", "application/x-subrip")
    second = await _upload(client, pid, "notes.srt", b"2\n", "application/x-subrip")
    third = await _upload(client, pid, "NOTES.srt", b"3\n", "application/x-subrip")
    assert [first["original_filename"], second["original_filename"],
            third["original_filename"]] == ["notes.srt", "notes_2.srt", "NOTES_3.srt"]
    # Stored object names are independent of the display name.
    assert second["filename"] == "notes.srt"

    # Another project is independent; a deleted Asset frees its name.
    other = await make_project("names-other")
    assert (await _upload(client, other, "notes.srt", b"x", "application/x-subrip"))[
        "original_filename"] == "notes.srt"
    assert (await client.delete(f"{API}/assets/{first['id']}")).status_code in (200, 204)
    again = await _upload(client, pid, "notes.srt", b"4\n", "application/x-subrip")
    assert again["original_filename"] == "notes.srt"


async def test_rename_to_a_taken_name_is_a_conflict(client: AsyncClient, make_project) -> None:
    pid = await make_project("names-rename")
    a = await _upload(client, pid, "a.srt", b"1\n", "application/x-subrip")
    b = await _upload(client, pid, "b.srt", b"2\n", "application/x-subrip")
    resp = await client.patch(f"{API}/assets/{b['id']}", json={"original_filename": "A.srt"})
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "conflict"
    assert (await client.get(f"{API}/assets/{b['id']}")).json()["original_filename"] == "b.srt"
    # Changing only the case of its own name is allowed.
    resp = await client.patch(f"{API}/assets/{a['id']}", json={"original_filename": "A.srt"})
    assert resp.status_code == 200 and resp.json()["original_filename"] == "A.srt"


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
async def test_image_outputs_carry_parameters_and_repeat_runs_are_numbered(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "test_image.png"
    subprocess.run(["convert", "-size", "2400x1600", "xc:teal", str(src)], check=True)
    pid = await make_project("names-image")
    asset = await _upload(client, pid, src.name, src.read_bytes(), "image/png")
    assert await _job(client, "image.resize", asset["id"], {"width": 1600}) == [
        "test_image_resize_1600x1067.png"]
    assert await _job(client, "image.resize", asset["id"], {"width": 1600}) == [
        "test_image_resize_1600x1067_2.png"]
    assert await _job(client, "image.crop", asset["id"],
                      {"x": 0, "y": 0, "width": 800, "height": 600}) == [
        "test_image_crop_800x600.png"]
    assert await _job(client, "image.format", asset["id"],
                      {"format": "webp", "quality": 80}) == ["test_image_convert_q80.webp"]
    assert await _job(client, "image.format", asset["id"], {"format": "jpeg"}) == [
        "test_image_convert.jpg"]
    assert await _job(client, "image.colour.adjust", asset["id"], {"brightness": 110}) == [
        "test_image_colour_b110-c0-s100.png"]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_video_and_audio_outputs_carry_parameters(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "video_main.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=duration=4:size=320x240:rate=10", "-f", "lavfi",
         "-i", "sine=duration=4", "-pix_fmt", "yuv420p", "-shortest", str(src)],
        check=True,
    )
    pid = await make_project("names-video")
    asset = await _upload(client, pid, src.name, src.read_bytes(), "video/mp4")
    aid = asset["id"]
    assert await _job(client, "video.trim", aid, {"start": 1, "end": 2.5}) == [
        "video_main_trim_1-2.5.mp4"]
    assert await _job(client, "video.trim", aid, {"start": 1}) == ["video_main_trim_1-end.mp4"]
    assert await _job(client, "video.trim", aid, {"start": 1}) == [
        "video_main_trim_1-end_2.mp4"]
    assert await _job(client, "video.transcode", aid, {
        "format": "mkv", "video_codec": "libx264", "quality": "compressed",
    }) == ["video_main_transcode_mkv-compressed.mkv"]
    assert await _job(client, "video.transcode", aid, {"format": "mp4"}) == [
        "video_main_transcode_mp4.mp4"]
    assert await _job(client, "video.resize", aid, {"width": 161}) == [
        "video_main_resize_160x120.mp4"]
    assert await _job(client, "video.crop", aid,
                      {"x": 0, "y": 0, "width": 160, "height": 120}) == [
        "video_main_crop_160x120.mp4"]
    assert await _job(client, "video.thumbnail", aid, {"timestamp": 1.5}) == [
        "video_main_thumb_1.5s.jpg"]
    assert await _job(client, "audio.gain", aid, {"gain_db": -3}) == [
        "video_main_gain_-3dB.mp4"]
    assert await _job(client, "audio.normalize", aid, {}) == [
        "video_main_normalized_-16LUFS.mp4"]
    assert await _job(client, "audio.fade", aid, {"fade_out": 1}) == [
        "video_main_fade_out1.mp4"]
    names = await _job(client, "video.split", aid, {"markers": [2]})
    assert len(set(names)) == 2
