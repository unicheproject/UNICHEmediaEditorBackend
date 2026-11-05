"""subtitle.autogenerate: timed SRT from provider transcription segments (offline)."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.capabilities.handlers.subtitle_autogenerate import Cue, parse_segments, to_srt
from app.core.config import settings
from app.core.database import async_session_factory
from app.core.errors import ProviderError
from app.providers.http import HTTPInferenceProvider
from app.services.media_metadata import probe_asset_metadata
from tests.test_provider_semantics import json_response, wire

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None

SEGMENTS = [
    {"id": 1, "start": 1.25, "end": 2.5, "text": "  Second   line. "},
    {"id": 0, "start": 0.0, "end": 1.25, "text": " First line."},
    {"id": 2, "start": 2.5, "end": 3.0, "text": "   "},  # blank: dropped
]


# --- pure helpers -----------------------------------------------------------


def test_segments_become_ordered_srt() -> None:
    cues = parse_segments(SEGMENTS)
    assert cues == [Cue(0.0, 1.25, "First line."), Cue(1.25, 2.5, "Second line.")]
    assert to_srt(cues) == (
        "1\n00:00:00,000 --> 00:00:01,250\nFirst line.\n\n"
        "2\n00:00:01,250 --> 00:00:02,500\nSecond line.\n"
    )
    assert to_srt([Cue(3661.5, 3662.0, "x")]).splitlines()[1] == (
        "01:01:01,500 --> 01:01:02,000"
    )


@pytest.mark.parametrize(
    "bad",
    [[{"start": 2, "end": 1, "text": "x"}], [{"start": -1, "end": 1, "text": "x"}],
     [{"start": 0, "end": "1", "text": "x"}], [{"start": 0, "end": 1}], ["x"],
     [{"start": True, "end": 1, "text": "x"}]],
)
def test_malformed_segments_are_rejected(bad: list) -> None:
    with pytest.raises(ProviderError, match="malformed"):
        parse_segments(bad)


def test_missing_segments_yield_no_cues() -> None:
    assert parse_segments(None) == []


def test_http_provider_routes_subtitles_to_the_transcription_path() -> None:
    provider = HTTPInferenceProvider(settings)
    assert provider._paths["subtitle.autogenerate"] == settings.inference_audio_transcribe_path


# --- job level (offline provider transport, real ffmpeg) ---------------------


def _av(path: Path, *, audio: bool = True) -> None:
    args = ["ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=160x120:rate=10"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest"]
    subprocess.run([*args, "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True)


async def _video_asset(client: AsyncClient, pid: str, path: Path) -> str:
    resp = await client.post(
        f"{API}/projects/{pid}/assets", files={"file": (path.name, path.read_bytes(), "video/mp4")}
    )
    assert resp.status_code == 201, resp.text
    asset_id: str = resp.json()["id"]
    async with async_session_factory() as session:
        await probe_asset_metadata(session, uuid.UUID(asset_id))
    return asset_id


async def _run(client: AsyncClient, asset_id: str, params: dict | None = None) -> dict:
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "subtitle.autogenerate", "asset_id": asset_id,
        "input": params or {},
    })
    assert resp.status_code == 201, resp.text
    job: dict = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    return job


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_video_gets_an_ai_generated_srt_asset(
    client: AsyncClient, make_project, monkeypatch, tmp_path: Path
) -> None:
    calls = wire(monkeypatch, json_response({
        "text": "First line. Second line.", "language": "english", "segments": SEGMENTS,
    }))
    monkeypatch.setattr(settings, "inference_provider", "openrouter")
    src = tmp_path / "interview.mp4"
    _av(src)
    pid = await make_project("subtitles")
    aid = await _video_asset(client, pid, src)

    job = await _run(client, aid, {"language": "en"})
    assert job["status"] == "succeeded", job
    output = job["output"]
    assert output["segment_count"] == 2 and output["language"] == "english"
    assert output["outputs"][0]["filename"] == "interview.srt"
    asset = (await client.get(f"{API}/assets/{output['outputs'][0]['asset_id']}")).json()
    assert asset["media_type"] == "subtitle"
    assert asset["ai_provenance"] == "generated"
    srt = (await client.get(output["outputs"][0]["download_path"])).text
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,250\nFirst line.")

    # One upload: the extracted mono M4A soundtrack, requesting timestamps.
    assert len(calls) == 1
    body = calls[0].content
    assert b'name="response_format"' in body and b"verbose_json" in body
    assert b'filename="interview_speech.m4a"' in body
    assert b'name="language"' in body


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize(
    ("response", "message"),
    [({"text": "Speech without timings."}, "no timed segments"),
     ({"text": "", "segments": []}, "No speech was detected")],
)
async def test_untimed_or_empty_transcripts_fail_without_an_asset(
    client: AsyncClient, make_project, monkeypatch, tmp_path: Path, response: dict, message: str
) -> None:
    wire(monkeypatch, json_response(response))
    monkeypatch.setattr(settings, "inference_provider", "openrouter")
    src = tmp_path / "clip.mp4"
    _av(src)
    pid = await make_project("subtitles-fail")
    job = await _run(client, await _video_asset(client, pid, src))
    assert job["status"] == "failed" and message in job["error"]
    assert job["output"] is None


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_silent_video_fails_before_calling_the_provider(
    client: AsyncClient, make_project, monkeypatch, tmp_path: Path
) -> None:
    calls = wire(monkeypatch, json_response({"text": "unused"}))
    monkeypatch.setattr(settings, "inference_provider", "openrouter")
    src = tmp_path / "silent.mp4"
    _av(src, audio=False)
    pid = await make_project("subtitles-silent")
    job = await _run(client, await _video_asset(client, pid, src))
    assert job["status"] == "failed" and "no sound" in job["error"]
    assert calls == []


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_audio_transcribe_output_is_unchanged(
    client: AsyncClient, make_project, monkeypatch, tmp_path: Path
) -> None:
    wire(monkeypatch, json_response({"text": "Hello.", "language": "english",
                                     "segments": SEGMENTS}))
    monkeypatch.setattr(settings, "inference_provider", "openrouter")
    src = tmp_path / "clip.mp4"
    _av(src)
    pid = await make_project("transcribe-unchanged")
    aid = await _video_asset(client, pid, src)
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "audio.transcribe", "asset_id": aid, "input": {},
    })
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded", job
    assert job["output"]["text"] == "Hello." and "segments" not in job["output"]
