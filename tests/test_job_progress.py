"""Duration-based Job progress from FFmpeg -progress output."""

from __future__ import annotations

import io
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.core.database import async_session_factory
from app.models.enums import JobStatus
from app.models.job import Job
from app.services import jobs as jobs_svc
from app.tools import progress, runner
from app.tools.demuxers import UNTRUSTED_INPUT
from tests.test_video_trim import _upload_video

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_CONVERT = shutil.which("convert") is not None


# --- parsing and tracker (no binaries) --------------------------------------


def test_parse_out_time_seconds() -> None:
    assert progress.parse_out_time_seconds("out_time_us=1500000\n") == 1.5
    assert progress.parse_out_time_seconds("out_time_ms=250000") == 0.25  # also µs
    assert progress.parse_out_time_seconds("out_time_us=N/A") is None
    assert progress.parse_out_time_seconds("out_time=00:00:01.500000") is None
    assert progress.parse_out_time_seconds("progress=end") is None
    assert progress.parse_out_time_seconds("out_time_us=-5") is None


async def test_tracker_accumulates_runs_caps_and_is_monotonic() -> None:
    seen: list[float] = []

    async def sink(fraction: float) -> None:
        seen.append(fraction)

    tracker = progress.ProgressTracker(10.0, sink)
    await tracker.update(2.5)
    await tracker.update(2.0)  # never goes backwards
    tracker.finish_run(5.0)
    await tracker.update(2.5)  # second run continues from 5 s
    await tracker.update(50.0)  # overshoot is capped below completion
    assert seen == [0.25, 0.75, 0.99]


async def test_failing_sink_never_raises() -> None:
    async def broken(_fraction: float) -> None:
        raise RuntimeError("database unavailable")

    await progress.ProgressTracker(4.0, broken).update(1.0)


async def test_tracking_needs_a_known_duration_and_resets() -> None:
    async def sink(_fraction: float) -> None:
        return None

    for unknown in (None, 0.0):
        with progress.tracking(unknown, sink) as active:
            assert active is False and progress.current() is None
    with progress.tracking(8.0, sink) as active:
        assert active is True
        progress.expect_output_seconds(2.0)
        tracker = progress.current()
        assert tracker is not None and tracker.expected_seconds == 2.0
    assert progress.current() is None


# --- runner with real FFmpeg ------------------------------------------------


def _make_video(path: Path, seconds: int) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x240:rate=25",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_runner_reports_ffmpeg_progress_only_when_tracking(
    tmp_path: Path, monkeypatch
) -> None:
    src = tmp_path / "in.mp4"
    _make_video(src, 6)
    launched: list[list[str]] = []
    original_exec = runner._exec

    async def spy(args, timeout, **kwargs):
        launched.append(list(args))
        return await original_exec(args, timeout, **kwargs)

    monkeypatch.setattr(runner, "_exec", spy)
    seen: list[float] = []

    async def sink(fraction: float) -> None:
        seen.append(fraction)

    # Every file input carries the demuxer allowlist (Task #18B), as in the app.
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *UNTRUSTED_INPUT, "-i", str(src),
           "-vf", "scale=160:120"]
    with progress.tracking(6.0, sink):
        await runner.run([*cmd, str(tmp_path / "tracked.mp4")])
    await runner.run([*cmd, str(tmp_path / "plain.mp4")])

    assert launched[0][1:4] == ["-progress", "pipe:1", "-nostats"]
    assert "-progress" not in launched[1]
    assert seen and seen == sorted(seen)
    assert seen[-1] == pytest.approx(0.99)  # the final block reports the full duration


# --- execute_job integration -----------------------------------------------


@pytest.fixture
def recorded_progress(monkeypatch) -> list[int]:
    """Record every progress value execute_job writes, without throttling."""
    monkeypatch.setattr(jobs_svc, "PROGRESS_MIN_INTERVAL_SECONDS", 0.0)
    values: list[int] = []
    original = jobs_svc._progress_sink

    def recording_sink(job_id: uuid.UUID):
        write = original(job_id)

        async def wrapped(fraction: float) -> None:
            values.append(
                jobs_svc.PROGRESS_START
                + int(fraction * (jobs_svc.PROGRESS_END - jobs_svc.PROGRESS_START))
            )
            await write(fraction)

        return wrapped

    monkeypatch.setattr(jobs_svc, "_progress_sink", recording_sink)
    return values


async def _run(client: AsyncClient, capability: str, asset_id: str, params: dict) -> dict:
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": capability, "asset_id": asset_id, "input": params,
    })
    assert resp.status_code == 201, resp.text
    job: dict = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded", job
    return job


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_video_job_reports_intermediate_progress(
    client: AsyncClient, make_project, tmp_path: Path, recorded_progress: list[int]
) -> None:
    src = tmp_path / "in.mp4"
    _make_video(src, 20)
    pid = await make_project("progress-resize")
    aid = await _upload_video(client, pid, src)
    job = await _run(client, "video.resize", aid, {"height": 120})
    assert recorded_progress, "no intermediate progress was reported"
    assert recorded_progress == sorted(recorded_progress)
    # The first FFmpeg report can land below one step (value 15); the real sink
    # drops it, but this recorder sees it, so the start value is allowed.
    assert all(jobs_svc.PROGRESS_START <= v < 100 for v in recorded_progress)
    assert recorded_progress[-1] == 94  # 0.99 of the 15..95 processing range
    assert job["progress"] == 100


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_trim_progress_follows_the_kept_window(
    client: AsyncClient, make_project, tmp_path: Path, recorded_progress: list[int]
) -> None:
    src = tmp_path / "in.mp4"
    _make_video(src, 20)
    pid = await make_project("progress-trim")
    aid = await _upload_video(client, pid, src)
    await _run(client, "video.trim", aid, {"start": 10})
    # Measured against 10 s of output, not the 20 s source (which would stop near 55).
    assert recorded_progress and recorded_progress[-1] == 94


@pytest.mark.skipif(not (HAVE_FFMPEG and HAVE_CONVERT), reason="ffmpeg/convert not installed")
async def test_jobs_without_known_duration_keep_the_midpoint_marker(
    client: AsyncClient, make_project, recorded_progress: list[int], monkeypatch, tmp_path: Path
) -> None:
    image = tmp_path / "pic.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=64x48", "-frames:v", "1", str(image)],
                   check=True, capture_output=True)
    pid = await make_project("progress-image")
    aid = (await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": ("pic.png", io.BytesIO(image.read_bytes()), "image/png")},
    )).json()["id"]
    seen_start: list[int] = []
    handler = jobs_svc.registry.get_handler("image.resize")
    original_run = handler.run

    async def observe(ctx):
        async with async_session_factory() as session:
            job = (await session.execute(
                Job.__table__.select().where(Job.asset_id == uuid.UUID(aid))
            )).one()
            seen_start.append(job.progress)
        assert progress.current() is None
        return await original_run(ctx)

    monkeypatch.setattr(handler, "run", observe)
    await _run(client, "image.resize", aid, {"width": 32, "height": 24})
    assert seen_start == [50]
    assert recorded_progress == []


async def test_progress_writes_are_fenced_to_running_jobs(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("progress-fence")
    async with async_session_factory() as session:
        job = Job(project_id=uuid.UUID(pid), capability_id="video.mute",
                  status=JobStatus.cancelled, input={}, progress=10)
        session.add(job)
        await session.commit()
        job_id = job.id
    await jobs_svc._progress_sink(job_id)(0.5)
    async with async_session_factory() as session:
        stored = await session.get(Job, job_id)
        assert stored is not None and stored.progress == 10
