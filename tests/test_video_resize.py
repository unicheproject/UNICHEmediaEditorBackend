"""video.resize: optional aspect-preserving side and even output sizes."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.agent.render import compile_execution_plan, render_group
from app.agent.reuse import REUSE_VERSIONS
from app.agent.schemas import Plan
from app.capabilities import registry
from app.capabilities.validation import (
    CapabilityInputValidationError,
    validate_capability_input,
)
from app.core.errors import ValidationError
from app.tools import ffmpeg
from tests.test_video_crop_thumbnail import _size
from tests.test_video_trim import _make_video, _upload_video

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
VID = "22222222-2222-2222-2222-222222222222"


async def _no_output_check(_output) -> None:
    """FFmpeg is mocked here and writes no file; skip the rendered-output check."""


def test_schema_needs_one_side_and_no_ffmpeg_sentinels() -> None:
    cap = registry.get("video.resize")
    for params in ({"height": 720}, {"width": 1280}, {"width": 1280, "height": 720}):
        validate_capability_input(cap, params)
    for params in ({}, {"width": -1, "height": 720}, {"width": 1}, {"height": 0}):
        with pytest.raises(CapabilityInputValidationError):
            validate_capability_input(cap, params)
    help_text = " ".join(p["description"] for p in cap.input_schema["properties"].values())
    assert "FFmpeg" not in help_text and "-1" not in help_text


def test_even_scale_dimensions() -> None:
    even = ffmpeg.even_scale_dimensions
    assert even(1280, 720) == (1280, 720)
    assert even(853, None) == (852, -2)
    assert even(None, 479) == (-2, 478)
    assert even(101, 51) == (100, 50)
    with pytest.raises(ValidationError, match="width, a height"):
        even(None, None)
    with pytest.raises(ValidationError, match="at least 2"):
        even(1, 720)


def test_reuse_versions_bumped_for_changed_output_semantics() -> None:
    assert REUSE_VERSIONS["video.resize"] == "2"
    assert REUSE_VERSIONS["video.crop"] == "2"


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (None, 120, (160, 120)),  # 320×240 source, aspect kept
        (160, None, (160, 120)),
        (101, 51, (100, 50)),  # previously failed in libx264
        (None, 179, (238, 178)),  # odd aspect result is made even
    ],
)
async def test_resize_outputs_even_sizes(
    tmp_path: Path, width: int | None, height: int | None, expected: tuple[int, int]
) -> None:
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _make_video(src)
    await ffmpeg.video_resize(str(src), str(dst), width, height)
    assert _size(dst) == expected


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_height_only_resize_job_succeeds(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "in.mp4"
    _make_video(src)
    pid = await make_project("resize-height")
    aid = await _upload_video(client, pid, src)
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "video.resize", "asset_id": aid, "input": {"height": 120},
    })
    assert resp.status_code == 201, resp.text
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded", job
    out = tmp_path / "out.mp4"
    out.write_bytes((await client.get(job["output"]["outputs"][0]["download_path"])).content)
    assert _size(out) == (160, 120)


async def test_agent_render_resize_with_one_side(monkeypatch, tmp_path) -> None:
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
    monkeypatch.setattr("app.agent.render._check_rendered_video", _no_output_check)
    plan = Plan.model_validate({"type": "plan", "steps": [
        {"id": "trim", "capability_id": "video.trim", "asset": VID,
         "params": {"start": 1, "end": 2}},
        {"id": "resize", "capability_id": "video.resize", "asset": "@trim",
         "params": {"height": 121}},
    ]})
    group = compile_execution_plan(plan, {VID: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "d" * 64,
        "technical_metadata_status": "pending", "has_audio_stream": None,
    }})["nodes"][0]
    await render_group(tmp_path / "in.mp4", tmp_path / "out.mp4", tmp_path, group)
    graph = calls[0][calls[0].index("-filter_complex") + 1]
    assert "scale=-2:120" in graph
