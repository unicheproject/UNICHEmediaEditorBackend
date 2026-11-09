"""MED/027: still-image clips (kind "image") in saved compositions."""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient
from pydantic import ValidationError as PydanticValidationError

from app.api.v1.jobs import get_enqueuer
from app.capabilities.compose_contract import (
    IMAGE_CLIP_MAX_DURATION_SECONDS,
    IMAGE_CLIP_MAX_PIXELS,
    IMAGE_CLIP_MIN_DURATION_SECONDS,
)
from app.composition.document import canonical_document
from app.core.database import async_session_factory
from app.core.errors import ToolError, ValidationError
from app.main import app
from app.models.asset import Asset
from app.schemas.composition import CompositionDocument
from app.services.jobs import execute_job
from app.services.media_metadata import StillImageInfo
from app.tools import compose_renderer, ffmpeg
from app.tools.compose_policy import LOSSLESS_COMPOSE_RENDER_POLICY
from tests.test_composition_clip_timing import NEED_FFMPEG, _clip_file, _durations, _put
from tests.test_composition_render import _compiled

API = "/api/v1"
A = "11111111-1111-1111-1111-111111111111"
B = "22222222-2222-2222-2222-222222222222"


def _video(asset_id: str, **fields: object) -> dict:
    return {"id": str(uuid.uuid4()), "asset_id": asset_id, **fields}


def _image(asset_id: str, duration: object = 2.0, **fields: object) -> dict:
    return {"id": str(uuid.uuid4()), "kind": "image", "asset_id": asset_id,
            "duration": duration, **fields}


def _document(clips: list[dict], audio_id: str | None = None) -> dict:
    return {
        "schema_version": 1,
        "canvas": {"width": 96, "height": 64},
        "tracks": [{"id": str(uuid.uuid4()), "kind": "video", "clips": clips}],
        "audio_bed": {"asset_id": audio_id, "mode": "mix", "volume": 0.5} if audio_id else None,
    }


def _canonical(raw: dict) -> tuple[dict, str]:
    return canonical_document(CompositionDocument.model_validate(raw))


# --- schema (no binaries) ---------------------------------------------------


def test_video_only_json_and_hash_are_unchanged() -> None:
    raw = _document([_video(A), _video(B, source_start=1.0, source_end=2.0)])
    data, digest = _canonical(raw)
    assert data == raw
    legacy = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert digest == hashlib.sha256(legacy.encode("utf-8")).hexdigest()


def test_explicit_video_kind_is_accepted_but_never_stored() -> None:
    raw = _document([_video(A)])
    plain = _canonical(raw)
    with_kind = json.loads(json.dumps(raw))
    with_kind["tracks"][0]["clips"][0]["kind"] = "video"
    explicit = _canonical(with_kind)
    assert explicit == plain
    assert "kind" not in explicit[0]["tracks"][0]["clips"][0]


def test_image_clip_serializes_in_order_with_video_clips() -> None:
    raw = _document([_video(A), _image(B, 5), _video(A, source_end=1.5)])
    data, _ = _canonical(raw)
    clips = data["tracks"][0]["clips"]
    assert clips[1] == {"id": raw["tracks"][0]["clips"][1]["id"], "kind": "image",
                        "asset_id": B, "duration": 5.0}
    assert "kind" not in clips[0] and clips[2]["source_end"] == 1.5


@pytest.mark.parametrize("clip", [
    _image(A, kind="audio"), _image(A, kind="IMAGE"), _image(A, kind=None), _image(A, kind=5),
    _image(A, source_start=1.0), _image(A, source_end=2.0),   # video timing on an image
    _video(A, duration=2.0),                                    # image duration on a video
    {"id": str(uuid.uuid4()), "kind": "image", "asset_id": A},  # missing duration
    _image(A, offset=1),
])
def test_wrong_or_ambiguous_clip_kinds_are_rejected(clip: dict) -> None:
    with pytest.raises(PydanticValidationError):
        CompositionDocument.model_validate(_document([clip]))


@pytest.mark.parametrize("duration", [
    0, -1, 0.02, IMAGE_CLIP_MIN_DURATION_SECONDS * 0.9, IMAGE_CLIP_MAX_DURATION_SECONDS + 0.01,
    math.inf, math.nan, "5", True, None,
])
def test_invalid_durations_are_rejected(duration: object) -> None:
    with pytest.raises(PydanticValidationError):
        CompositionDocument.model_validate(_document([_image(A, duration)]))


@pytest.mark.parametrize("duration", [IMAGE_CLIP_MIN_DURATION_SECONDS, 1, 600])
def test_duration_bounds_are_inclusive(duration: float) -> None:
    CompositionDocument.model_validate(_document([_image(A, duration)]))


def test_pixel_limit_is_100_megapixels() -> None:
    assert IMAGE_CLIP_MAX_PIXELS == 100_000_000
    ffmpeg.check_still_image_pixels(10_000, 10_000)
    ffmpeg.check_still_image_pixels(None, None)  # unknown: checked again at render
    with pytest.raises(ValidationError, match="100 megapixels"):
        ffmpeg.check_still_image_pixels(10_001, 10_000)


# --- save, compile and resolve -------------------------------------------------


async def _upload(client: AsyncClient, pid: str, name: str, data: bytes = b"x") -> str:
    response = await client.post(
        f"{API}/projects/{pid}/assets", files={"file": (name, io.BytesIO(data), "image/png")}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def test_saved_mixed_composition_compiles_in_order(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("image-clips-compile")
    video = await _upload(client, pid, "clip.mp4")
    image = await _upload(client, pid, "photo.png")
    saved = await _put(client, pid, _document(
        [_video(video), _image(image, 1.5), _video(video, source_start=0.5)]
    ))
    assert saved.status_code == 201, saved.text
    got = (await client.get(
        f"{API}/projects/{pid}/composition/revisions/{saved.json()['revision_id']}"
    )).json()["document"]["tracks"][0]["clips"]
    assert [c.get("kind") for c in got] == [None, "image", None]
    request = (await _compiled(saved.json()["revision_id"])).request
    assert [str(i) for i in request.video_asset_ids] == [video, image, video]
    assert request.still_durations == (None, 1.5, None)
    assert request.clip_windows is not None and request.clip_windows[2].start == 0.5


async def test_video_only_composition_has_no_still_durations(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("image-clips-legacy")
    video = await _upload(client, pid, "clip.mp4")
    saved = (await _put(client, pid, _document([_video(video)]))).json()
    request = (await _compiled(saved["revision_id"])).request
    assert request.still_durations is None and request.clip_windows is None


@pytest.mark.parametrize("swap", ["image_as_video", "video_as_image", "both_kinds"])
async def test_media_type_mismatches_are_rejected_on_save(
    client: AsyncClient, make_project, swap: str
) -> None:
    pid = await make_project(f"image-clips-mismatch-{swap}")
    video = await _upload(client, pid, "clip.mp4")
    image = await _upload(client, pid, "photo.png")
    clips = {
        "image_as_video": [_video(image)],
        "video_as_image": [_image(video)],
        "both_kinds": [_image(image), _video(image)],
    }[swap]
    response = await _put(client, pid, _document(clips))
    assert response.status_code == 422, response.text
    assert (await client.get(f"{API}/projects/{pid}/composition")).status_code == 404


@pytest.mark.parametrize(("size", "accepted"), [((10_000, 10_000), True), ((12_000, 9_000), False)])
async def test_pixel_limit_is_checked_on_save_when_known(
    client: AsyncClient, make_project, size: tuple[int, int], accepted: bool
) -> None:
    pid = await make_project(f"image-clips-pixels-{size}")
    image = await _upload(client, pid, "huge.png")
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(image))
        assert asset is not None
        asset.technical_metadata_status = "available"
        asset.width, asset.height = size
        await session.commit()
    response = await _put(client, pid, _document([_image(image)]))
    assert response.status_code == (201 if accepted else 422), response.text
    if not accepted:
        assert "megapixels" in response.json()["error"]["message"]


async def test_pixel_limit_is_checked_before_rendering(monkeypatch, tmp_path: Path) -> None:
    async def huge(_path: Path) -> StillImageInfo:
        return StillImageInfo(20_000, 20_000, "TopLeft", False)

    async def no_ffmpeg(*_args, **_kwargs) -> str:
        raise AssertionError("FFmpeg must not decode an oversized image")

    monkeypatch.setattr(ffmpeg, "probe_still_image", huge)
    monkeypatch.setattr(ffmpeg.runner, "run", no_ffmpeg)
    with pytest.raises(ValidationError, match="too large"):
        await ffmpeg.compose(["/img.png"], str(tmp_path / "o.mp4"), width=96, height=64,
                             work_dir=tmp_path, still_durations=[2.0])


async def test_renderer_passes_still_durations_only_when_set(monkeypatch, tmp_path) -> None:
    from app.composition.render import ComposeRenderRequest

    calls: list[dict] = []

    async def fake_compose(_segments, _dst, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(compose_renderer.ffmpeg, "compose", fake_compose)
    sources = compose_renderer.ComposeRenderSources(video_paths=("/a", "/b"))
    ids = (uuid.UUID(A), uuid.UUID(B))
    await compose_renderer.render_compose(
        ComposeRenderRequest(ids, 96, 64), sources, "/o.mp4", work_dir=tmp_path
    )
    await compose_renderer.render_compose(
        ComposeRenderRequest(ids, 96, 64, still_durations=(None, 2.0)),
        sources, "/o.mp4", work_dir=tmp_path,
    )
    assert "still_durations" not in calls[0]
    assert calls[1]["still_durations"] == [None, 2.0]


# --- real rendering (tiny fixtures) -------------------------------------------


def _convert(*args: str) -> None:
    subprocess.run(["convert", *args], check=True, capture_output=True, timeout=30)


def _with_exif_orientation(jpeg: bytes, orientation: int) -> bytes:
    entry = struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0)
    tiff = b"MM\x00*" + struct.pack(">I", 8) + struct.pack(">H", 1) + entry + b"\x00" * 4
    payload = b"Exif\x00\x00" + tiff
    return jpeg[:2] + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + jpeg[2:]


def _pixel(path: Path, seconds: float, x: int, y: int) -> tuple[int, int, int]:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True, timeout=30,
    ).stdout
    index = (y * 96 + x) * 3
    return raw[index], raw[index + 1], raw[index + 2]


def _mean_volume(path: Path, start: float, length: float) -> float:
    err = subprocess.run(
        ["ffmpeg", "-v", "info", "-ss", str(start), "-t", str(length), "-i", str(path),
         "-af", "volumedetect", "-vn", "-f", "null", "-"],
        capture_output=True, text=True, timeout=30,
    ).stderr
    line = next(row for row in err.splitlines() if "mean_volume" in row)
    return float(line.split("mean_volume:")[1].split("dB")[0])


def _streams(path: Path) -> tuple[set[str], float]:
    data = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name:format=duration",
         "-of", "json", str(path)], check=True, capture_output=True, timeout=30,
    ).stdout)
    return {s["codec_name"] for s in data["streams"]}, float(data["format"]["duration"])


def _is(colour: tuple[int, int, int], name: str) -> bool:
    r, g, b = colour
    return {
        "red": r > 150 and g < 90 and b < 90,
        "blue": b > 150 and r < 90 and g < 90,
        "black": r < 40 and g < 40 and b < 40,
    }[name]


@NEED_FFMPEG
async def test_mixed_clips_keep_order_duration_and_silence(tmp_path: Path) -> None:
    red = _clip_file(tmp_path / "red.mp4", "red", 2)
    blue = tmp_path / "blue.png"
    _convert("-size", "96x64", "xc:blue", str(blue))
    out = tmp_path / "out.mp4"
    await ffmpeg.compose(
        [str(red), str(blue), str(red)], str(out), width=96, height=64, work_dir=tmp_path,
        clip_windows=[(0.0, 1.0), (0.0, None), (0.0, 0.5)],
        still_durations=[None, 1.5, None],
    )
    durations = _durations(out)
    assert durations["video"] == pytest.approx(3.0, abs=2 / 30)
    assert durations["audio"] == pytest.approx(durations["video"], abs=0.1)
    assert _is(_pixel(out, 0.5, 48, 32), "red")
    assert _is(_pixel(out, 1.7, 48, 32), "blue")
    assert _is(_pixel(out, 2.8, 48, 32), "red")
    assert _mean_volume(out, 0.2, 0.6) > -40      # source soundtrack
    assert _mean_volume(out, 1.3, 1.0) < -80      # image clip: silence
    assert _mean_volume(out, 2.6, 0.3) > -40


@pytest.mark.parametrize("duration", [IMAGE_CLIP_MIN_DURATION_SECONDS, 0.5, 2.5])
@NEED_FFMPEG
async def test_image_duration_is_frame_accurate(tmp_path: Path, duration: float) -> None:
    image = tmp_path / "i.png"
    _convert("-size", "96x64", "xc:blue", str(image))
    out = tmp_path / "out.mp4"
    await ffmpeg.compose([str(image)], str(out), width=96, height=64, work_dir=tmp_path,
                         still_durations=[duration])
    frames = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    assert int(frames) == round(duration * 30)


@NEED_FFMPEG
async def test_audio_bed_covers_image_clips_in_sync(tmp_path: Path) -> None:
    image = tmp_path / "i.png"
    _convert("-size", "96x64", "xc:blue", str(image))
    bed = tmp_path / "bed.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=5",
                    "-c:a", "aac", str(bed)], check=True, capture_output=True, timeout=30)
    out = tmp_path / "out.mp4"
    await ffmpeg.compose([str(image), str(image)], str(out), width=96, height=64,
                         work_dir=tmp_path, audio_path=str(bed), audio_mode="mix",
                         music_volume=0.5, still_durations=[1.0, 1.0])
    durations = _durations(out)
    assert durations["video"] == pytest.approx(2.0, abs=2 / 30)
    assert durations["audio"] == pytest.approx(durations["video"], abs=0.1)
    assert _mean_volume(out, 0.2, 1.5) > -40


@pytest.mark.parametrize("container", ["jpeg-exif", "tiff-tag"])
@NEED_FFMPEG
async def test_orientation_6_is_displayed_upright(tmp_path: Path, container: str) -> None:
    # Orientation 6 (RightTop) turns the 60×40 halves into a 40×60 portrait with
    # red on top; on the 96×64 canvas it is pillarboxed in black.
    if container == "jpeg-exif":
        plain = tmp_path / "plain.jpg"
        _convert("-size", "30x40", "xc:red", "-size", "30x40", "xc:blue", "+append", str(plain))
        image = tmp_path / "rotated.jpg"
        image.write_bytes(_with_exif_orientation(plain.read_bytes(), 6))
    else:
        image = tmp_path / "rotated.tif"
        _convert("-size", "30x40", "xc:red", "-size", "30x40", "xc:blue", "+append",
                 "-orient", "RightTop", str(image))
    out = tmp_path / "out.mp4"
    await ffmpeg.compose([str(image)], str(out), width=96, height=64, work_dir=tmp_path,
                         still_durations=[0.5])
    assert _is(_pixel(out, 0.1, 48, 10), "red")
    assert _is(_pixel(out, 0.1, 48, 54), "blue")
    assert _is(_pixel(out, 0.1, 4, 32), "black")
    assert _is(_pixel(out, 0.1, 91, 32), "black")


@NEED_FFMPEG
async def test_transparency_is_composited_onto_black(tmp_path: Path) -> None:
    # Left: fully transparent pixels whose stored colour is white; right: 50% red.
    image = tmp_path / "alpha.png"
    _convert("-size", "48x64", "xc:white", "-alpha", "set", "-channel", "A", "-evaluate",
             "set", "0", "+channel", "(", "-size", "48x64", "xc:rgba(255,0,0,0.5)", ")",
             "+append", f"PNG32:{image}")
    out = tmp_path / "out.mp4"
    await ffmpeg.compose([str(image)], str(out), width=96, height=64, work_dir=tmp_path,
                         still_durations=[0.5])
    assert _is(_pixel(out, 0.1, 20, 32), "black")
    r, g, b = _pixel(out, 0.1, 75, 32)
    assert 100 <= r <= 160 and g < 40 and b < 40


@NEED_FFMPEG
async def test_default_and_lossless_policies(tmp_path: Path) -> None:
    image = tmp_path / "i.png"
    _convert("-size", "96x64", "xc:blue", str(image))
    default = tmp_path / "out.mp4"
    await ffmpeg.compose([str(image)], str(default), width=96, height=64,
                         work_dir=tmp_path, still_durations=[1.0])
    assert _streams(default)[0] == {"h264", "aac"}
    lossless = tmp_path / f"out{LOSSLESS_COMPOSE_RENDER_POLICY.suffix}"
    work = tmp_path / "lossless"
    work.mkdir()
    await ffmpeg.compose([str(image)], str(lossless), width=96, height=64, work_dir=work,
                         still_durations=[1.0], policy=LOSSLESS_COMPOSE_RENDER_POLICY)
    codecs, seconds = _streams(lossless)
    assert codecs == {"ffv1", "flac"} and seconds == pytest.approx(1.0, abs=2 / 30)


@NEED_FFMPEG
async def test_unreadable_image_fails_safely(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ToolError, match="cannot be read"):
        await ffmpeg.compose([str(bad)], str(tmp_path / "o.mp4"), width=96, height=64,
                             work_dir=tmp_path, still_durations=[1.0])


async def _export(client: AsyncClient, pid: str, revision_id: str):
    async def leave_queued(_job_id: uuid.UUID) -> None:
        return None

    app.dependency_overrides[get_enqueuer] = lambda: leave_queued
    job = (await client.post(f"{API}/jobs", json={
        "capability_id": "project.export", "project_id": pid,
        "input": {"composition_revision_id": revision_id, "preset": "web"},
    })).json()
    async with async_session_factory() as session:
        return await execute_job(session, uuid.UUID(job["id"]))


@NEED_FFMPEG
async def test_export_with_unreadable_image_fails_without_an_asset(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("image-clips-export-fail")
    image = await _upload(client, pid, "broken.png", b"not an image")
    saved = await _put(client, pid, _document([_image(image)]))
    assert saved.status_code == 201, saved.text
    before = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    executed = await _export(client, pid, saved.json()["revision_id"])
    assert executed.status.value == "failed" and executed.output is None
    assert "Composition image cannot be read" in (executed.error or "")
    after = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    assert after == before


@NEED_FFMPEG
async def test_export_renders_video_and_image_clips(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid = await make_project("image-clips-export")
    red = _clip_file(tmp_path / "red.mp4", "red", 1)
    video = await _upload(client, pid, "red.mp4", red.read_bytes())
    png = tmp_path / "blue.png"
    _convert("-size", "96x64", "xc:blue", str(png))
    image = await _upload(client, pid, "blue.png", png.read_bytes())
    saved = (await _put(client, pid, _document([_video(video), _image(image, 1.0)]))).json()
    executed = await _export(client, pid, saved["revision_id"])
    assert executed.status.value == "succeeded", executed.error
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(executed.output["outputs"][0]["asset_id"]))
        assert asset is not None and asset.duration_seconds is not None
        assert asset.duration_seconds == pytest.approx(2.0, abs=0.15)
        assert asset.provenance_kind == "derived"
