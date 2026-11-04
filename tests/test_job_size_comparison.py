"""Job output size_comparison: informational input/output byte comparison."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import AsyncClient

from app.capabilities import registry
from app.models.enums import MediaType
from app.services.jobs import _size_comparison
from tests.test_video_trim import _make_video, _upload_video

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def _job(asset_id: uuid.UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(asset_id=asset_id)


def _ctx(size: int | None, media_type: str = "video") -> SimpleNamespace:
    return SimpleNamespace(input_asset_meta={"media_type": media_type, "size_bytes": size})


def _out(size: int, media_type: MediaType = MediaType.video) -> SimpleNamespace:
    return SimpleNamespace(size_bytes=size, media_type=media_type)


def test_larger_output_is_reported_with_positive_delta() -> None:
    result = _size_comparison(_job(uuid.uuid4()), _ctx(1000), [_out(1280)])  # type: ignore[arg-type]
    assert result == {
        "input_size_bytes": 1000, "output_size_bytes": 1280,
        "delta_bytes": 280, "ratio": 1.28,
    }


def test_split_segments_are_summed() -> None:
    result = _size_comparison(
        _job(uuid.uuid4()), _ctx(1000), [_out(300), _out(300), _out(250)]  # type: ignore[arg-type]
    )
    assert result is not None
    assert (result["output_size_bytes"], result["delta_bytes"]) == (850, -150)


@pytest.mark.parametrize(
    ("job", "ctx", "outputs"),
    [
        (_job(None), _ctx(1000), [_out(10)]),  # no primary input (e.g. concat)
        (_job(uuid.uuid4()), _ctx(None), [_out(10)]),  # unknown input size
        (_job(uuid.uuid4()), _ctx(0), [_out(10)]),
        (_job(uuid.uuid4()), _ctx(1000), [_out(10, MediaType.image)]),  # thumbnail
        (_job(uuid.uuid4()), _ctx(1000), []),  # JSON-only result
    ],
)
def test_comparison_omitted_when_not_meaningful(job, ctx, outputs) -> None:
    assert _size_comparison(job, ctx, outputs) is None


def test_file_output_schema_declares_size_comparison() -> None:
    schema = registry.get("video.transcode").output_schema
    declared = schema["properties"]["size_comparison"]
    assert set(declared["required"]) == {
        "input_size_bytes", "output_size_bytes", "delta_bytes", "ratio",
    }


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_jobs_report_comparison_only_for_same_media_outputs(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "in.mp4"
    _make_video(src)
    pid = await make_project("size-comparison")
    aid = await _upload_video(client, pid, src)
    source_size = (await client.get(f"{API}/assets/{aid}")).json()["size_bytes"]

    async def run(capability: str, params: dict) -> dict:
        resp = await client.post(f"{API}/jobs", json={
            "capability_id": capability, "asset_id": aid, "input": params,
        })
        assert resp.status_code == 201, resp.text
        job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
        assert job["status"] == "succeeded", job
        output: dict = job["output"]
        return output

    resized = await run("video.resize", {"height": 120})
    comparison = resized["size_comparison"]
    output_size = resized["outputs"][0]["size_bytes"]
    assert comparison == {
        "input_size_bytes": source_size,
        "output_size_bytes": output_size,
        "delta_bytes": output_size - source_size,
        "ratio": round(output_size / source_size, 4),
    }

    thumbnail = await run("video.thumbnail", {"timestamp": 1})
    assert "size_comparison" not in thumbnail
