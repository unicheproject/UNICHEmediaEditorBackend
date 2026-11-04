"""audio.trim interval validation: schema, admission and execution."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.agent.reuse import REUSE_VERSIONS
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


def test_schema_and_cross_field_rules() -> None:
    cap = registry.get("audio.trim")
    assert cap.input_schema["required"] == ["start"]
    validate_capability_input(cap, {"start": 2})
    for params in ({"start": -1}, {"start": 3, "end": 3}, {"start": 3, "end": 1}):
        with pytest.raises(CapabilityInputValidationError):
            validate_capability_input(cap, params)


def test_trim_reuse_versions_bumped() -> None:
    assert REUSE_VERSIONS["audio.trim"] == "2"
    assert REUSE_VERSIONS["video.trim"] == "2"


def test_resolver_uses_the_media_label() -> None:
    with pytest.raises(ValidationError, match="end of the audio"):
        ffmpeg.resolve_trim_window(9, None, duration=4, frame_rate=None, media="audio")


def _make_audio(path: Path, seconds: int = 4) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}", str(path)],
        check=True, capture_output=True,
    )


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


async def _upload(client: AsyncClient, pid: str, src: Path, *, probe: bool = True) -> str:
    resp = await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": ("tone.wav", src.read_bytes(), "audio/wav")},
    )
    assert resp.status_code == 201, resp.text
    asset_id: str = resp.json()["id"]
    if probe:
        async with async_session_factory() as session:
            await probe_asset_metadata(session, uuid.UUID(asset_id))
    return asset_id


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize(
    "params", [{"start": 1}, {"start": 1, "end": 30}, {"start": 1, "end": 2.5}]
)
async def test_trim_keeps_requested_window(
    client: AsyncClient, make_project, tmp_path: Path, params: dict
) -> None:
    src = tmp_path / "tone.wav"
    _make_audio(src)
    pid = await make_project("audio-trim")
    aid = await _upload(client, pid, src)
    resp = await client.post(
        f"{API}/jobs", json={"capability_id": "audio.trim", "asset_id": aid, "input": params}
    )
    assert resp.status_code == 201, resp.text
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded", job
    out = tmp_path / "out.wav"
    out.write_bytes((await client.get(job["output"]["outputs"][0]["download_path"])).content)
    expected = min(params.get("end", 4), 4) - params["start"]
    assert _duration(out) == pytest.approx(expected, abs=0.05)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_admission_rejects_start_after_end_of_audio(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "tone.wav"
    _make_audio(src)
    pid = await make_project("audio-trim-bounds")
    aid = await _upload(client, pid, src)
    resp = await client.post(
        f"{API}/jobs", json={"capability_id": "audio.trim", "asset_id": aid, "input": {"start": 9}}
    )
    assert resp.status_code == 422, resp.text
    assert "before the end of the audio" in resp.json()["error"]["message"]
    assert (await client.get(f"{API}/projects/{pid}/jobs")).json()["total"] == 0


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_execution_revalidates_and_cleans_up(tmp_path: Path) -> None:
    src, dst = tmp_path / "tone.wav", tmp_path / "out.wav"
    _make_audio(src)
    with pytest.raises(ValidationError, match="end of the audio"):
        await ffmpeg.audio_trim(str(src), str(dst), 9)
    assert not dst.exists()
    await ffmpeg.audio_trim(str(src), str(dst), 0.5, 1.5)
    assert _duration(dst) == pytest.approx(1.0, abs=0.05)
