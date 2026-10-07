"""Canonical asset technical metadata probing, persistence and Agent context."""

import asyncio
import io
import json
import shutil
import subprocess
import uuid
from types import SimpleNamespace

import pytest
from httpx import AsyncClient

from app.agent import llm
from app.api.v1.assets import get_asset_probe_enqueuer
from app.core.database import async_session_factory
from app.main import app
from app.models.enums import MediaType, TechnicalMetadataStatus
from app.scripts import backfill_asset_metadata
from app.services.agent import _planning_asset_context
from app.services.assets import create_derived_asset
from app.services.media_metadata import _parse_ffprobe, probe_asset_metadata, probe_media

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_IMAGEMAGICK = shutil.which("convert") is not None and shutil.which("identify") is not None


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(not HAVE_IMAGEMAGICK, reason="ImageMagick is not installed")
async def test_image_probe_persists_dimensions_and_normalized_stream(tmp_path) -> None:
    image = tmp_path / "fixture.png"
    subprocess.run(
        ["convert", "-size", "24x16", "xc:#336699", str(image)],
        check=True,
        capture_output=True,
    )
    result = await probe_media(image, MediaType.image)
    assert result.status == TechnicalMetadataStatus.available
    assert (result.width, result.height) == (24, 16)
    assert result.container_format == "PNG"
    assert result.technical_streams == {
        "streams": [{"index": 0, "type": "image", "codec": "PNG", "width": 24, "height": 16}],
        "streams_truncated": False,
    }
    assert "storage_path" not in str(result.technical_streams)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="FFmpeg is not installed")
async def test_audio_video_and_silent_video_probe_summaries(tmp_path) -> None:
    audio = tmp_path / "tone.wav"
    with_audio = tmp_path / "video-audio.mp4"
    silent = tmp_path / "video-silent.mp4"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "pcm_s16le", str(audio))
    for destination, inputs in (
        (
            with_audio,
            [
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:size=64x48:rate=10:duration=1",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-c:v",
                "mpeg4",
                "-c:a",
                "aac",
                "-shortest",
            ],
        ),
        (
            silent,
            ["-f", "lavfi", "-i", "color=c=red:size=64x48:rate=10:duration=1", "-c:v", "mpeg4"],
        ),
    ):
        _ffmpeg(*inputs, str(destination))

    audio_result = await probe_media(audio, MediaType.audio)
    av_result = await probe_media(with_audio, MediaType.video)
    silent_result = await probe_media(silent, MediaType.video)

    assert audio_result.has_audio_stream is True
    assert audio_result.has_video_stream is False
    assert audio_result.audio_sample_rate == 44100
    assert audio_result.duration_seconds == pytest.approx(1, abs=0.2)
    assert av_result.has_video_stream is True and av_result.has_audio_stream is True
    assert (av_result.width, av_result.height) == (64, 48)
    assert av_result.video_codec and av_result.audio_codec
    assert av_result.frame_rate == pytest.approx(10, abs=0.1)
    assert av_result.technical_streams and len(av_result.technical_streams["streams"]) == 2
    assert {stream["index"] for stream in av_result.technical_streams["streams"]} == {0, 1}
    assert silent_result.has_video_stream is True
    assert silent_result.has_audio_stream is False
    assert silent_result.audio_codec is None


def test_multistream_summary_is_bounded_but_preserves_presence_facts() -> None:
    streams = [
        {"index": index, "codec_type": "subtitle", "codec_name": "subrip"} for index in range(64)
    ]
    streams.append(
        {
            "index": 64,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 320,
            "height": 240,
            "avg_frame_rate": "25/1",
        }
    )
    result = _parse_ffprobe(json.dumps({"streams": streams, "format": {}}).encode())
    assert result.has_video_stream is True
    assert result.width == 320 and result.height == 240
    assert result.technical_streams is not None
    assert len(result.technical_streams["streams"]) == 64
    assert result.technical_streams["streams_truncated"] is True


@pytest.mark.skipif(not HAVE_IMAGEMAGICK, reason="ImageMagick is not installed")
async def test_upload_is_pending_and_readable_until_queued_probe_finishes(
    client: AsyncClient, make_project, tmp_path
) -> None:
    image = tmp_path / "upload.png"
    subprocess.run(["convert", "-size", "20x12", "xc:green", str(image)], check=True)
    queued: list[uuid.UUID] = []

    async def capture(asset_id: uuid.UUID) -> None:
        queued.append(asset_id)

    app.dependency_overrides[get_asset_probe_enqueuer] = lambda: capture
    project_id = await make_project("metadata-upload")
    with image.open("rb") as stream:
        response = await client.post(
            f"{API}/projects/{project_id}/assets",
            files={"file": (image.name, stream, "image/png")},
        )
    assert response.status_code == 201, response.text
    asset_id = uuid.UUID(response.json()["id"])
    assert response.json()["technical_metadata_status"] == "pending"
    assert queued == [asset_id]

    pending = await client.get(f"{API}/assets/{asset_id}")
    assert pending.status_code == 200
    assert pending.json()["technical_metadata_status"] == "pending"
    assert pending.json()["width"] is None

    async with async_session_factory() as session:
        result = await probe_asset_metadata(session, asset_id)
    assert result == TechnicalMetadataStatus.available
    available = await client.get(f"{API}/assets/{asset_id}")
    assert available.json()["technical_metadata_status"] == "available"
    assert (available.json()["width"], available.json()["height"]) == (20, 12)


async def test_probe_failure_marks_metadata_unavailable_but_asset_remains_readable(
    client: AsyncClient, make_project
) -> None:
    project_id = await make_project("metadata-corrupt")
    uploaded = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("broken.mp4", io.BytesIO(b"not media"), "video/mp4")},
    )
    assert uploaded.status_code == 201
    asset_id = uuid.UUID(uploaded.json()["id"])
    async with async_session_factory() as session:
        result = await probe_asset_metadata(session, asset_id)
    assert result == TechnicalMetadataStatus.unavailable
    response = await client.get(f"{API}/assets/{asset_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["technical_metadata_status"] == "unavailable"
    assert body["duration_seconds"] is None
    assert "stderr" not in response.text.lower()


@pytest.mark.skipif(not shutil.which("python3"), reason="Python is not installed")
async def test_probe_timeout_is_controlled_and_unavailable(tmp_path, monkeypatch) -> None:
    from app.services import media_metadata

    media = tmp_path / "input.wav"
    media.write_bytes(b"not actually parsed by the fake probe")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_probe = fake_bin / "ffprobe"
    fake_probe.write_text("#!/usr/bin/python3\nimport time\ntime.sleep(10)\n")
    fake_probe.chmod(0o755)
    spawn = asyncio.create_subprocess_exec
    async def fixture_spawn(*args, **kwargs):
        return await spawn("/usr/bin/python3", str(fake_probe), **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fixture_spawn)
    monkeypatch.setattr(media_metadata, "PROBE_TIMEOUT_SECONDS", 0.05)

    result = await media_metadata.probe_media_safely(media, MediaType.audio)
    assert result.status == TechnicalMetadataStatus.unavailable
    assert result.technical_streams is None


@pytest.mark.skipif(not HAVE_IMAGEMAGICK, reason="ImageMagick is not installed")
async def test_derived_asset_is_probed_before_persistence(make_project, tmp_path) -> None:
    image = tmp_path / "derived.png"
    subprocess.run(["convert", "-size", "30x18", "xc:purple", str(image)], check=True)
    project_id = uuid.UUID(await make_project("metadata-derived"))
    async with async_session_factory() as session:
        asset = await create_derived_asset(
            session,
            project_id=project_id,
            source_asset_id=None,
            src_path=image,
            filename="derived.png",
            media_type=MediaType.image,
        )
    assert asset.technical_metadata_status == TechnicalMetadataStatus.available.value
    assert (asset.width, asset.height) == (30, 18)


@pytest.mark.skipif(not HAVE_IMAGEMAGICK, reason="ImageMagick is not installed")
async def test_backfill_skips_available_assets_and_continues_after_bad_media(
    client: AsyncClient, make_project, tmp_path, monkeypatch
) -> None:
    image = tmp_path / "backfill.png"
    subprocess.run(["convert", "-size", "10x8", "xc:orange", str(image)], check=True)
    project_id = await make_project("metadata-backfill")
    with image.open("rb") as stream:
        uploaded = await client.post(
            f"{API}/projects/{project_id}/assets",
            files={"file": (image.name, stream, "image/png")},
        )
    corrupt = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("bad.png", io.BytesIO(b"broken"), "image/png")},
    )
    assert uploaded.status_code == corrupt.status_code == 201

    counts = await backfill_asset_metadata.backfill(project_id=uuid.UUID(project_id), batch_size=1)
    assert counts == {"available": 1, "unavailable": 1, "skipped": 0}

    original_probe = backfill_asset_metadata.probe_asset_metadata
    touched: list[uuid.UUID] = []

    async def count_probe(session, asset_id):
        touched.append(asset_id)
        return await original_probe(session, asset_id)

    monkeypatch.setattr(backfill_asset_metadata, "probe_asset_metadata", count_probe)
    second_counts = await backfill_asset_metadata.backfill(
        project_id=uuid.UUID(project_id), batch_size=1
    )
    assert second_counts["available"] == 0
    assert len(touched) == 1  # corrupt/unavailable is retried; successful is skipped


def test_agent_context_includes_known_metadata_and_hides_untrusted_fields() -> None:
    known = SimpleNamespace(
        media_type=MediaType.video,
        technical_metadata_status=TechnicalMetadataStatus.available.value,
        duration_seconds=12.5,
        width=1920,
        height=1080,
        has_video_stream=True,
        has_audio_stream=False,
        video_codec="h264",
        audio_codec=None,
        frame_rate=25.0,
        audio_sample_rate=None,
        audio_channels=None,
        storage_path="secret/path.mp4",
        technical_streams={"raw": "should not appear"},
    )
    context = _planning_asset_context(known)  # type: ignore[arg-type]
    assert context["duration_seconds"] == 12.5
    assert context["has_audio_stream"] is False
    assert "storage_path" not in context
    assert "technical_streams" not in context

    pending = SimpleNamespace(**{**known.__dict__, "technical_metadata_status": "pending"})
    unknown = _planning_asset_context(pending)  # type: ignore[arg-type]
    assert unknown["duration_seconds"] is None
    assert unknown["has_video_stream"] is None


async def test_openrouter_prompt_receives_compact_asset_context(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeBackend:
        def __init__(self, settings) -> None:
            pass

        async def generate(self, messages: list[dict]) -> dict:
            captured.extend(messages)
            return {"content": '{"type":"clarification","question":"How long?","missing":[]}'}

    monkeypatch.setattr(llm, "OpenRouterBackend", FakeBackend)
    settings = llm.Settings(agent_provider="openrouter", openrouter_api_key="test")
    await llm.propose(
        [],
        "Edit this clip",
        {"asset-id": "video"},
        asset_context={"asset-id": {"media_type": "video", "duration_seconds": 12.5}},
        settings=settings,
    )
    prompt = next(
        message["content"] for message in captured if "IN-SCOPE ASSETS" in message["content"]
    )
    assert '"duration_seconds":12.5' in prompt
    assert "secret/path" not in prompt
