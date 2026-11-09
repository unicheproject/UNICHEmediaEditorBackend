"""MED/027/MED-014: optional per-clip source in/out points (source_start/source_end)."""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient
from pydantic import ValidationError as PydanticValidationError

from app.api.v1.jobs import get_enqueuer
from app.composition.document import canonical_document
from app.composition.render import ClipWindow
from app.core.database import async_session_factory
from app.core.errors import ToolError, ValidationError
from app.main import app
from app.models.asset import Asset
from app.schemas.composition import CompositionDocument
from app.services.jobs import execute_job
from app.tools import compose_renderer, ffmpeg
from app.tools.compose_policy import DEFAULT_COMPOSE_RENDER_POLICY
from tests.test_composition_render import _compiled

API = "/api/v1"
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
NEED_FFMPEG = pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg + ffprobe")
A = "11111111-1111-1111-1111-111111111111"
B = "22222222-2222-2222-2222-222222222222"


def _clip(asset_id: str, **timing: float) -> dict:
    return {"id": str(uuid.uuid4()), "asset_id": asset_id, **timing}


def _document(clips: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "canvas": {"width": 96, "height": 64},
        "tracks": [{"id": str(uuid.uuid4()), "kind": "video", "clips": clips}],
        "audio_bed": None,
    }


# --- schema (no binaries) ---------------------------------------------------


def test_untimed_document_serializes_and_hashes_exactly_as_before() -> None:
    raw = _document([_clip(A), _clip(B)])
    data, digest = canonical_document(CompositionDocument.model_validate(raw))
    assert data == raw  # no source_start/source_end keys appear for legacy clips
    legacy = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert digest == hashlib.sha256(legacy.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(("timing", "expected"), [
    ({"source_start": 1, "source_end": 2.5}, {"source_start": 1.0, "source_end": 2.5}),
    ({"source_start": 0.5}, {"source_start": 0.5}),
    ({"source_end": 3}, {"source_end": 3.0}),
    ({"source_start": 0}, {"source_start": 0.0}),
])
def test_valid_timing_round_trips(timing: dict, expected: dict) -> None:
    clip = _clip(A, **timing)
    data, _ = canonical_document(CompositionDocument.model_validate(_document([clip])))
    stored = data["tracks"][0]["clips"][0]
    assert {k: v for k, v in stored.items() if k in {"source_start", "source_end"}} == expected
    # Timing is part of the revision identity.
    assert canonical_document(CompositionDocument.model_validate(_document([clip])))[1] != (
        canonical_document(CompositionDocument.model_validate(_document([_clip(A)])))[1]
    )


@pytest.mark.parametrize("timing", [
    {"source_start": -1}, {"source_end": 0}, {"source_end": -2},
    {"source_start": 3, "source_end": 3}, {"source_start": 4, "source_end": 2},
    {"source_start": math.inf}, {"source_start": math.nan}, {"source_end": math.inf},
    {"source_start": "1"}, {"source_start": True},
    {"source_end": None, "source_start": None, "offset": 1},
    # The timeline-ambiguous names are not accepted as aliases.
    {"start": 1}, {"end": 2},
])
def test_invalid_timing_is_rejected(timing: dict) -> None:
    with pytest.raises(PydanticValidationError):
        CompositionDocument.model_validate(_document([_clip(A, **timing)]))


def test_openapi_exposes_source_timing_fields() -> None:
    schemas = app.openapi()["components"]["schemas"]
    clip = schemas["CompositionClip"]
    assert clip["required"] == ["id", "asset_id"]
    # `kind` ("video" or omitted) was added with image clips; it is never stored.
    assert set(clip["properties"]) == {"id", "kind", "asset_id", "source_start", "source_end"}
    start = clip["properties"]["source_start"]["anyOf"][0]
    end = clip["properties"]["source_end"]["anyOf"][0]
    assert (start["type"], start["minimum"]) == ("number", 0)
    assert (end["type"], end["exclusiveMinimum"]) == ("number", 0)


# --- API persistence and save-time checks ------------------------------------


async def _upload(client: AsyncClient, project_id: str, name: str, data: bytes) -> str:
    response = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": (name, io.BytesIO(data), "video/mp4")},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _put(client: AsyncClient, project_id: str, document: dict):
    return await client.put(
        f"{API}/projects/{project_id}/composition", json=document,
        headers={"If-None-Match": "*"},
    )


async def test_timing_persists_and_legacy_clips_stay_untimed(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("clip-timing-persist")
    first = await _upload(client, pid, "a.mp4", b"a")
    second = await _upload(client, pid, "b.mp4", b"b")
    document = _document([
        _clip(first, source_start=1.5, source_end=4), _clip(second), _clip(first, source_end=2),
    ])
    saved = await _put(client, pid, document)
    assert saved.status_code == 201, saved.text
    revision_id = saved.json()["revision_id"]

    got = (await client.get(
        f"{API}/projects/{pid}/composition/revisions/{revision_id}"
    )).json()["document"]["tracks"][0]["clips"]
    assert [(c["asset_id"], c.get("source_start"), c.get("source_end")) for c in got] == [
        (first, 1.5, 4.0), (second, None, None), (first, None, 2.0),
    ]
    assert "source_start" not in got[1] and "source_end" not in got[1]

    compiled = await _compiled(revision_id)
    assert compiled.request.clip_windows == (
        ClipWindow(1.5, 4.0), ClipWindow(0.0, None), ClipWindow(0.0, 2.0),
    )
    assert [str(i) for i in compiled.request.video_asset_ids] == [first, second, first]


async def test_untimed_composition_compiles_without_windows(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("clip-timing-legacy")
    first = await _upload(client, pid, "a.mp4", b"a")
    saved = (await _put(client, pid, _document([_clip(first)]))).json()
    assert (await _compiled(saved["revision_id"])).request.clip_windows is None


@pytest.mark.parametrize(("timing", "message"), [
    ({"source_start": 12}, "start must be before the end"),
    ({"source_start": 1, "source_end": 1.01}, "at least one video frame"),
])
async def test_save_rejects_ranges_outside_probed_source(
    client: AsyncClient, make_project, timing: dict, message: str
) -> None:
    pid = await make_project(f"clip-timing-bounds-{timing}")
    aid = await _upload(client, pid, "a.mp4", b"a")
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(aid))
        assert asset is not None
        asset.technical_metadata_status = "available"
        asset.duration_seconds, asset.frame_rate = 10.0, 25.0
        await session.commit()
    response = await _put(client, pid, _document([_clip(aid, **timing)]))
    assert response.status_code == 422
    assert message in response.json()["error"]["message"]
    assert (await client.get(f"{API}/projects/{pid}/composition")).status_code == 404

async def test_renderer_passes_windows_only_when_set(monkeypatch, tmp_path) -> None:
    from app.composition.render import ComposeRenderRequest

    calls: list[dict] = []

    async def fake_compose(segments, dst, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(compose_renderer.ffmpeg, "compose", fake_compose)
    sources = compose_renderer.ComposeRenderSources(video_paths=("/a", "/b"))
    ids = (uuid.UUID(A), uuid.UUID(B))
    await compose_renderer.render_compose(
        ComposeRenderRequest(ids, 96, 64), sources, "/o.mp4", work_dir=tmp_path
    )
    timed = ComposeRenderRequest(ids, 96, 64, clip_windows=(ClipWindow(1, 2), ClipWindow()))
    await compose_renderer.render_compose(timed, sources, "/o.mp4", work_dir=tmp_path)
    assert "clip_windows" not in calls[0]
    assert calls[1]["clip_windows"] == [(1, 2), (0.0, None)]
    assert calls[1]["policy"] == DEFAULT_COMPOSE_RENDER_POLICY
    with pytest.raises(ValidationError, match="clip timing"):
        await compose_renderer.render_compose(
            ComposeRenderRequest(ids, 96, 64, clip_windows=(ClipWindow(),)),
            sources, "/o.mp4", work_dir=tmp_path,
        )


# --- real rendering (tiny fixtures) -------------------------------------------


def _clip_file(path: Path, colour: str, seconds: float, *, audio_seconds: float | None = None):
    audio = audio_seconds if audio_seconds is not None else seconds
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"color=c={colour}:s=96x64:r=10:d={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={audio}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True, capture_output=True, timeout=30,
    )
    return path


def _durations(path: Path) -> dict[str, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
         "-of", "json", str(path)], check=True, capture_output=True, timeout=30,
    ).stdout
    return {s["codec_type"]: float(s["duration"]) for s in json.loads(out)["streams"]}


def _colour_at(path: Path, seconds: float) -> tuple[int, int, int]:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(path), "-frames:v", "1",
         "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True, timeout=30,
    ).stdout
    return raw[0], raw[1], raw[2]


@NEED_FFMPEG
async def test_timed_clips_render_in_order_with_synced_audio(tmp_path: Path) -> None:
    red = _clip_file(tmp_path / "red.mp4", "red", 2)
    blue = _clip_file(tmp_path / "blue.mp4", "blue", 2)
    out = tmp_path / "out.mp4"
    await ffmpeg.compose(
        [str(red), str(blue), str(red)], str(out), width=96, height=64, work_dir=tmp_path,
        clip_windows=[(0.5, 1.5), (1.0, None), (0.0, 0.5)],  # in+out, in only, out only
    )
    durations = _durations(out)
    assert durations["video"] == pytest.approx(2.5, abs=0.15)
    assert durations["audio"] == pytest.approx(durations["video"], abs=0.1)
    r, g, b = _colour_at(out, 0.5)
    assert r > 150 and b < 100  # first clip: red
    r, g, b = _colour_at(out, 1.5)
    assert b > 150 and r < 100  # second clip: blue
    r, g, b = _colour_at(out, 2.2)
    assert r > 150 and b < 100  # third clip: red again


@NEED_FFMPEG
async def test_untimed_compose_is_unchanged(tmp_path: Path) -> None:
    red = _clip_file(tmp_path / "red.mp4", "red", 1)
    out = tmp_path / "out.mp4"
    await ffmpeg.compose([str(red), str(red)], str(out), width=96, height=64, work_dir=tmp_path)
    assert _durations(out)["video"] == pytest.approx(2.0, abs=0.15)


@NEED_FFMPEG
async def test_range_past_the_last_video_frame_is_rejected(tmp_path: Path) -> None:
    # 1 s of video in a 3 s container: the video stream, not the container, bounds it.
    src = _clip_file(tmp_path / "short.mp4", "red", 1, audio_seconds=3)
    with pytest.raises(ValidationError, match="start must be before the end"):
        await ffmpeg.compose(
            [str(src)], str(tmp_path / "out.mp4"), width=96, height=64, work_dir=tmp_path,
            clip_windows=[(2.0, 2.5)],
        )


@NEED_FFMPEG
async def test_frameless_clip_segment_fails(monkeypatch, tmp_path: Path) -> None:
    src = _clip_file(tmp_path / "red.mp4", "red", 2)

    async def ffmpeg_exit_0_without_frames(args, **_kwargs) -> str:
        Path(args[-1]).write_bytes(b"")
        return ""

    monkeypatch.setattr(ffmpeg.runner, "run", ffmpeg_exit_0_without_frames)
    with pytest.raises(ToolError, match="does not contain video frames"):
        await ffmpeg.compose(
            [str(src)], str(tmp_path / "out.mp4"), width=96, height=64, work_dir=tmp_path,
            clip_windows=[(0.5, 1.0)],
        )


@NEED_FFMPEG
async def test_export_with_invalid_render_time_range_fails_without_an_asset(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid = await make_project("clip-timing-export-fail")
    src = _clip_file(tmp_path / "red.mp4", "red", 1)
    aid = await _upload(client, pid, "red.mp4", src.read_bytes())
    # Metadata is still pending, so save cannot check the range; the render must.
    saved = await _put(client, pid, _document([_clip(aid, source_start=5)]))
    assert saved.status_code == 201, saved.text

    async def leave_queued(_job_id: uuid.UUID) -> None:
        return None

    app.dependency_overrides[get_enqueuer] = lambda: leave_queued
    job = (await client.post(f"{API}/jobs", json={
        "capability_id": "project.export", "project_id": pid,
        "input": {"composition_revision_id": saved.json()["revision_id"], "preset": "web"},
    })).json()
    before = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    async with async_session_factory() as session:
        executed = await execute_job(session, uuid.UUID(job["id"]))
    assert executed.status.value == "failed"
    assert "start must be before the end" in (executed.error or "")
    assert executed.output is None
    after = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    assert after == before


@NEED_FFMPEG
async def test_export_renders_a_timed_clip(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid = await make_project("clip-timing-export")
    src = _clip_file(tmp_path / "red.mp4", "red", 2)
    aid = await _upload(client, pid, "red.mp4", src.read_bytes())
    clip = _clip(aid, source_start=0.5, source_end=1.5)
    saved = (await _put(client, pid, _document([clip]))).json()

    async def leave_queued(_job_id: uuid.UUID) -> None:
        return None

    app.dependency_overrides[get_enqueuer] = lambda: leave_queued
    job = (await client.post(f"{API}/jobs", json={
        "capability_id": "project.export", "project_id": pid,
        "input": {"composition_revision_id": saved["revision_id"], "preset": "web"},
    })).json()
    async with async_session_factory() as session:
        executed = await execute_job(session, uuid.UUID(job["id"]))
        assert executed.status.value == "succeeded", executed.error
        asset = await session.get(Asset, uuid.UUID(executed.output["outputs"][0]["asset_id"]))
        assert asset is not None and asset.duration_seconds is not None
        assert asset.duration_seconds == pytest.approx(1.0, abs=0.15)



@NEED_FFMPEG
async def test_timed_clip_under_the_lossless_policy(tmp_path: Path) -> None:
    from app.tools.compose_policy import LOSSLESS_COMPOSE_RENDER_POLICY

    red = _clip_file(tmp_path / "red.mp4", "red", 2)
    out = tmp_path / f"out{LOSSLESS_COMPOSE_RENDER_POLICY.suffix}"
    await ffmpeg.compose(
        [str(red)], str(out), width=96, height=64, work_dir=tmp_path,
        clip_windows=[(0.5, 1.5)], policy=LOSSLESS_COMPOSE_RENDER_POLICY,
    )
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name:format=duration",
         "-of", "json", str(out)], check=True, capture_output=True, timeout=30,
    ).stdout)
    assert {s["codec_name"] for s in probe["streams"]} == {"ffv1", "flac"}
    assert float(probe["format"]["duration"]) == pytest.approx(1.0, abs=0.15)
