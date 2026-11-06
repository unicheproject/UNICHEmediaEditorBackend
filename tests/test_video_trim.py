"""video.trim interval validation: schema, admission, execution and Agent render."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.agent.render import compile_execution_plan, render_group
from app.agent.schemas import Plan
from app.capabilities import registry
from app.capabilities.validation import (
    CapabilityInputValidationError,
    validate_capability_input,
)
from app.core.database import async_session_factory
from app.core.errors import ValidationError
from app.services.media_metadata import probe_asset_metadata
from app.tools import ffmpeg

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
VID = "22222222-2222-2222-2222-222222222222"


def _trim_cap():
    return registry.get("video.trim")


# --- schema + cross-field rules (no binaries) -------------------------------


def test_schema_makes_end_optional_and_bounds_start() -> None:
    schema = _trim_cap().input_schema
    assert schema["required"] == ["start"]
    assert schema["properties"]["start"]["minimum"] == 0
    assert schema["properties"]["end"]["exclusiveMinimum"] == 0
    validate_capability_input(_trim_cap(), {"start": 3})


@pytest.mark.parametrize(
    "params",
    [{"start": -1}, {"start": 0, "end": 0}, {"start": 5, "end": 5}, {"start": 5, "end": 2}],
)
def test_invalid_intervals_rejected_by_canonical_validation(params: dict) -> None:
    with pytest.raises(CapabilityInputValidationError):
        validate_capability_input(_trim_cap(), params)


def test_agent_plan_validation_shares_the_interval_rule() -> None:
    with pytest.raises(CapabilityInputValidationError, match="greater than input.start"):
        validate_capability_input(_trim_cap(), {"start": 4, "end": 1}, allow_step_refs=True)


async def test_end_before_start_returns_422_before_job_creation(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("trim-order")
    aid = (await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": ("clip.mp4", io.BytesIO(b"not media"), "video/mp4")},
    )).json()["id"]
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "video.trim", "asset_id": aid, "input": {"start": 2, "end": 1},
    })
    assert resp.status_code == 422
    assert "greater than input.start" in resp.json()["error"]["message"]
    jobs = (await client.get(f"{API}/projects/{pid}/jobs")).json()
    assert jobs["total"] == 0


# --- source-dependent window resolution (pure) ------------------------------


def test_resolve_window_clamps_and_rejects() -> None:
    resolve = ffmpeg.resolve_trim_window
    assert resolve(1, None, duration=10, frame_rate=25) is None
    assert resolve(1, 10, duration=10, frame_rate=25) is None  # end at duration
    assert resolve(1, 99, duration=10, frame_rate=25) is None  # end past duration
    assert resolve(1, 4.5, duration=10, frame_rate=25) == 4.5
    assert resolve(0, 0.04, duration=10, frame_rate=25) == 0.04  # exactly one frame
    # Unknown timing defers bounds to output validation.
    assert resolve(50, None, duration=None, frame_rate=None) is None
    with pytest.raises(ValidationError, match="before the end"):
        resolve(10, None, duration=10, frame_rate=25)
    with pytest.raises(ValidationError, match="one video frame"):
        resolve(0, 0.001, duration=10, frame_rate=25)
    with pytest.raises(ValidationError, match="one video frame"):
        resolve(9.99, None, duration=10, frame_rate=25)
    with pytest.raises(ValidationError, match="after the start"):
        resolve(3, 3, duration=None, frame_rate=None)


# --- execution (needs ffmpeg) -----------------------------------------------


def _make_video(path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


async def _upload_video(
    client: AsyncClient, pid: str, src: Path, *, probe: bool = True
) -> str:
    resp = await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": ("clip.mp4", src.read_bytes(), "video/mp4")},
    )
    assert resp.status_code == 201, resp.text
    asset_id: str = resp.json()["id"]
    if probe:  # API tests leave uploads pending unless they opt in.
        async with async_session_factory() as session:
            await probe_asset_metadata(session, uuid.UUID(asset_id))
        detail = (await client.get(f"{API}/assets/{asset_id}")).json()
        assert detail["technical_metadata_status"] == "available"
    return asset_id


async def _download_output(client: AsyncClient, job: dict, tmp_path: Path) -> Path:
    assert job["status"] == "succeeded", job
    output = job["output"]["outputs"][0]
    body = (await client.get(f"{API}/assets/{output['asset_id']}/download")).content
    path = tmp_path / "out.mp4"
    path.write_bytes(body)
    return path


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize("params", [{"start": 1}, {"start": 1, "end": 30}])
async def test_trim_keeps_from_start_to_end_of_video(
    client: AsyncClient, make_project, tmp_path: Path, params: dict
) -> None:
    src = tmp_path / "in.mp4"
    _make_video(src)
    pid = await make_project("trim-onward")
    aid = await _upload_video(client, pid, src)
    resp = await client.post(
        f"{API}/jobs", json={"capability_id": "video.trim", "asset_id": aid, "input": params}
    )
    assert resp.status_code == 201, resp.text
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    out = await _download_output(client, job, tmp_path)
    assert _duration(out) == pytest.approx(1.0, abs=0.1)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"start": 0, "end": 0.001}, "one video frame"),
        ({"start": 5}, "before the end of the video"),
        ({"start": 2, "end": 9}, "before the end of the video"),
    ],
)
async def test_admission_rejects_windows_outside_probed_source(
    client: AsyncClient, make_project, tmp_path: Path, params: dict, message: str
) -> None:
    src = tmp_path / "in.mp4"
    _make_video(src)
    pid = await make_project("trim-bounds")
    aid = await _upload_video(client, pid, src)
    resp = await client.post(
        f"{API}/jobs", json={"capability_id": "video.trim", "asset_id": aid, "input": params}
    )
    assert resp.status_code == 422, resp.text
    assert message in resp.json()["error"]["message"]
    assert (await client.get(f"{API}/projects/{pid}/jobs")).json()["total"] == 0


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_pending_metadata_defers_bounds_to_execution(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "in.mp4"
    _make_video(src)
    pid = await make_project("trim-pending")
    aid = await _upload_video(client, pid, src, probe=False)
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "video.trim", "asset_id": aid, "input": {"start": 5},
    })
    assert resp.status_code == 201, resp.text
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "failed"
    assert "before the end of the video" in job["error"]
    assert job["output"] is None


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_execution_revalidates_source_when_admission_could_not(tmp_path: Path) -> None:
    # Admission skips bounds while metadata is pending; the tool probes itself.
    src = tmp_path / "in.mp4"
    _make_video(src)
    dst = tmp_path / "out.mp4"
    with pytest.raises(ValidationError, match="one video frame"):
        await ffmpeg.video_trim(str(src), str(dst), 0, 0.001)
    with pytest.raises(ValidationError, match="before the end"):
        await ffmpeg.video_trim(str(src), str(dst), 7)
    assert not dst.exists()

    await ffmpeg.video_trim(str(src), str(dst), 0.5, 1.5)
    assert _duration(dst) == pytest.approx(1.0, abs=0.1)


# --- Agent render path ------------------------------------------------------


async def test_agent_render_trim_without_end_omits_end_bound(monkeypatch, tmp_path) -> None:
    from app.tools import runner

    calls: list[list[str]] = []

    async def no_audio(_path: str) -> bool:
        return False

    async def capture(args: list[str], **_kwargs) -> str:
        calls.append(args)
        return ""

    async def source_timing(_path, _media_type):
        from app.models.enums import TechnicalMetadataStatus
        from app.services.media_metadata import TechnicalMetadata

        return TechnicalMetadata(
            TechnicalMetadataStatus.available, duration_seconds=240.0, frame_rate=25.0,
            has_video_stream=True,
        )

    monkeypatch.setattr(ffmpeg, "probe_media", source_timing)
    monkeypatch.setattr(ffmpeg, "_has_audio", no_audio)
    monkeypatch.setattr(runner, "run", capture)
    plan = Plan.model_validate({"type": "plan", "steps": [
        {"id": "trim", "capability_id": "video.trim", "asset": VID, "params": {"start": 3}},
        {"id": "resize", "capability_id": "video.resize", "asset": "@trim",
         "params": {"width": 80, "height": 60}},
    ]})
    group = compile_execution_plan(plan, {VID: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "d" * 64,
        "technical_metadata_status": "pending", "has_audio_stream": None,
    }})["nodes"][0]
    await render_group(tmp_path / "in.mp4", tmp_path / "out.mp4", tmp_path, group)
    graph = calls[0][calls[0].index("-filter_complex") + 1]
    assert "trim=start=3,setpts" in graph
    assert ":end=" not in graph
