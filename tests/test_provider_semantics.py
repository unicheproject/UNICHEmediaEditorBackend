"""Offline wire/handler/Job contracts: transport success is not inference success."""

from __future__ import annotations

import asyncio
import io
import shutil
import subprocess
import uuid
import wave
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select

from app.capabilities import registry
from app.capabilities.context import JobContext
from app.capabilities.handlers import base as handler_base
from app.capabilities.readiness import resolve_availability
from app.core.config import settings
from app.core.database import async_session_factory
from app.core.errors import ProviderError
from app.models.asset import Asset
from app.models.enums import JobStatus
from app.providers.base import BaseInferenceProvider, InferenceRequest
from app.providers.http import HTTPInferenceProvider
from app.providers.openrouter import (
    AUDIO_BYTES_KEY,
    AUDIO_FILENAME_KEY,
    OpenRouterInferenceProvider,
)
from app.providers.validation import require_success_object
from app.schemas.job import JobCreate
from app.services import jobs as jobs_svc

FAILURES = [
    {"status": "error"},
    {"status": "failed"},
    {"status": "not_implemented"},
    {"status": " ERROR "},
    {"success": False},
    {"ok": False},
    {"error": "provider failed"},
    {"error": {"message": "provider failed"}},
]
WRONG_TYPES = [None, 7, [], {}, True]
TOOLS = shutil.which("ffprobe") and shutil.which("ffmpeg")


def wire(monkeypatch, response=None, *, error=None, gate=None):
    """Exercise real httpx decoding, without any network or provider secrets."""
    client = httpx.AsyncClient
    calls = []

    async def receive(request):
        calls.append(request)
        if gate:
            gate[0].set()
            await gate[1].wait()
        if error:
            raise error("synthetic-sensitive-diagnostic", request=request)
        return response

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(
            transport=httpx.MockTransport(receive),
            **kwargs,
        ),
    )
    monkeypatch.setattr(settings, "inference_base_url", "http://inference.test")
    monkeypatch.setattr(settings, "inference_api_key", "")
    monkeypatch.setattr(settings, "openrouter_base_url", "http://openrouter.test")
    monkeypatch.setattr(settings, "openrouter_api_key", "synthetic-test-only")
    return calls


def json_response(value):
    return httpx.Response(200, json=value)


def audio_response(body, mime="audio/mpeg"):
    return httpx.Response(200, content=body, headers={"content-type": mime})


@pytest.mark.parametrize("failure", FAILURES)
def test_explicit_failure_wins_over_success_fields(failure):
    with pytest.raises(ProviderError, match="reported failure"):
        require_success_object({"caption": "Hello", "text": "", **failure})


@pytest.mark.parametrize(
    "extra",
    [
        {"message": "provider message"},
        {"detail": "provider detail"},
        {"status": "unknown"},
        {"success": 0},
        {"ok": 0},
        {"error": ""},
        {"error": None},
    ],
)
def test_unknown_fields_and_nonboolean_flags_are_not_failure(extra):
    assert require_success_object({"caption": "Hello", **extra})["caption"] == "Hello"


@pytest.mark.parametrize("cap,field", [("image.caption", "caption"), ("audio.transcribe", "text")])
@pytest.mark.parametrize("payload", [{}, *FAILURES])
async def test_http_handler_rejects_missing_result_and_explicit_failure(
    monkeypatch, cap, field, payload
):
    wire(monkeypatch, json_response(payload))
    provider = HTTPInferenceProvider(settings)
    monkeypatch.setattr(handler_base, "get_provider", lambda: provider)
    with pytest.raises(ProviderError):
        await registry.get_handler(cap).run(JobContext(capability_id=cap))


@pytest.mark.parametrize("cap,field", [("image.caption", "caption"), ("audio.transcribe", "text")])
@pytest.mark.parametrize("value", WRONG_TYPES)
async def test_http_handler_rejects_wrong_result_type(monkeypatch, cap, field, value):
    wire(monkeypatch, json_response({field: value}))
    monkeypatch.setattr(handler_base, "get_provider", lambda: HTTPInferenceProvider(settings))
    with pytest.raises(ProviderError):
        await registry.get_handler(cap).run(JobContext(capability_id=cap))


@pytest.mark.parametrize("value", ["", "   "])
async def test_caption_rejects_blank(monkeypatch, value):
    wire(monkeypatch, json_response({"caption": value}))
    monkeypatch.setattr(handler_base, "get_provider", lambda: HTTPInferenceProvider(settings))
    with pytest.raises(ProviderError):
        await registry.get_handler("image.caption").run(JobContext(capability_id="image.caption"))


@pytest.mark.parametrize(
    "cap,field,value",
    [
        ("image.caption", "caption", "Hello world"),
        ("audio.transcribe", "text", "Hello world"),
        ("audio.transcribe", "text", ""),
    ],
)
@pytest.mark.parametrize("extra", [{}, {"extra": "allowed", "message": "informational"}])
async def test_http_valid_success_preserves_fields(monkeypatch, cap, field, value, extra):
    wire(monkeypatch, json_response({field: value, **extra}))
    monkeypatch.setattr(handler_base, "get_provider", lambda: HTTPInferenceProvider(settings))
    result = await registry.get_handler(cap).run(JobContext(capability_id=cap))
    assert result.data == {field: value, **extra, "provider": "http"}


@pytest.mark.parametrize("cap,field", [("image.caption", "caption"), ("audio.transcribe", "text")])
@pytest.mark.parametrize("failure", FAILURES)
async def test_http_valid_result_with_failure_is_rejected(monkeypatch, cap, field, failure):
    wire(monkeypatch, json_response({field: "Hello", **failure}))
    monkeypatch.setattr(handler_base, "get_provider", lambda: HTTPInferenceProvider(settings))
    with pytest.raises(ProviderError):
        await registry.get_handler(cap).run(JobContext(capability_id=cap))


@pytest.mark.parametrize("status", [201, 302, 400, 502])
async def test_http_status_is_strict_2xx_and_errors_do_not_echo_body(monkeypatch, status):
    response = httpx.Response(status, json={"caption": "Hello", "detail": "secret-html-marker"})
    wire(monkeypatch, response)
    provider = HTTPInferenceProvider(settings)
    request = InferenceRequest(capability_id="image.caption")
    if status == 201:
        assert (await provider.infer(request))["caption"] == "Hello"
    else:
        with pytest.raises(ProviderError) as error:
            await provider.infer(request)
        assert str(status) in str(error.value)
        assert "secret-html-marker" not in str(error.value)
        assert len(str(error.value)) < 100


@pytest.mark.parametrize("provider_name", ["http", "openrouter"])
@pytest.mark.parametrize("body", [b"null", b"[]", b'"text"', b"{not-json", b"\xff"])
async def test_json_shape_and_syntax_fail_controlled(monkeypatch, tmp_path, provider_name, body):
    wire(monkeypatch, httpx.Response(200, content=body))
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"input")
    provider = (
        HTTPInferenceProvider(settings)
        if provider_name == "http"
        else OpenRouterInferenceProvider(settings)
    )
    with pytest.raises(ProviderError):
        await provider.infer(
            InferenceRequest(capability_id="audio.transcribe", file_path=str(audio))
        )


@pytest.mark.parametrize(
    "provider_name,cap",
    [
        ("http", "image.caption"),
        ("openrouter", "audio.transcribe"),
        ("openrouter", "audio.tts"),
    ],
)
@pytest.mark.parametrize("exception", [httpx.ReadTimeout, httpx.ConnectError])
async def test_transport_failures_are_sanitized(
    monkeypatch, tmp_path, provider_name, cap, exception
):
    wire(monkeypatch, error=exception)
    source = tmp_path / "source.wav"
    source.write_bytes(b"input")
    provider = (
        HTTPInferenceProvider(settings)
        if provider_name == "http"
        else OpenRouterInferenceProvider(settings)
    )
    with pytest.raises(ProviderError) as error:
        await provider.infer(
            InferenceRequest(capability_id=cap, payload={"text": "Hello"}, file_path=str(source))
        )
    assert "synthetic-sensitive-diagnostic" not in str(error.value)
    assert "http://" not in str(error.value)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        *FAILURES,
        *[{"text": value} for value in WRONG_TYPES],
        *[{"text": "Hello", **failure} for failure in FAILURES],
    ],
)
async def test_openrouter_transcript_validates_raw_body(monkeypatch, tmp_path, payload):
    wire(monkeypatch, json_response(payload))
    source = tmp_path / "source.wav"
    source.write_bytes(b"input")
    with pytest.raises(ProviderError):
        await OpenRouterInferenceProvider(settings).infer(
            InferenceRequest(
                capability_id="audio.transcribe",
                file_path=str(source),
            )
        )


@pytest.mark.parametrize("text", ["Hello world", ""])
async def test_openrouter_explicit_text_and_metadata_preserved(monkeypatch, tmp_path, text):
    wire(monkeypatch, json_response({"text": text, "language": "en", "extra": "allowed"}))
    source = tmp_path / "source.wav"
    source.write_bytes(b"input")
    result = await OpenRouterInferenceProvider(settings).infer(
        InferenceRequest(
            capability_id="audio.transcribe",
            file_path=str(source),
        )
    )
    assert result["text"] == text and result["language"] == "en"


@pytest.mark.parametrize("cap", ["audio.tts", "audio.transcribe"])
@pytest.mark.parametrize("status", [302, 403, 503])
async def test_openrouter_non2xx_is_sanitized(monkeypatch, tmp_path, cap, status):
    wire(monkeypatch, httpx.Response(status, text="<html>synthetic-secret</html>"))
    source = tmp_path / "source.wav"
    source.write_bytes(b"input")
    with pytest.raises(ProviderError) as error:
        await OpenRouterInferenceProvider(settings).infer(
            InferenceRequest(
                capability_id=cap,
                file_path=str(source),
                payload={"text": "Hello"},
            )
        )
    assert str(status) in str(error.value)
    assert "synthetic-secret" not in str(error.value) and "<html>" not in str(error.value)


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x00\x00" * 2400)
    return buffer.getvalue()


async def persisted_job(client, make_project, cap):
    pid = uuid.UUID(await make_project("Provider semantics"))
    aid = None
    if cap != "audio.tts":
        name, mime = (
            ("source.png", "image/png") if cap == "image.caption" else ("source.wav", "audio/wav")
        )
        aid = uuid.UUID(
            (
                await client.post(
                    f"/api/v1/projects/{pid}/assets",
                    files={
                        "file": (name, b"input", mime),
                    },
                )
            ).json()["id"]
        )
    async with async_session_factory() as session:
        job = await jobs_svc.create_job(
            session,
            JobCreate(
                capability_id=cap,
                project_id=pid,
                asset_id=aid,
                input={"text": "Hello"} if cap == "audio.tts" else {},
            ),
        )
        return job.id


@pytest.mark.parametrize(
    "cap,provider",
    [
        ("image.caption", "http"),
        ("audio.transcribe", "http"),
        ("audio.transcribe", "openrouter"),
    ],
)
async def test_semantic_failure_job_fails_without_outputs_and_redelivery_is_noop(
    client,
    make_project,
    monkeypatch,
    cap,
    provider,
):
    calls = wire(monkeypatch, json_response({"status": "not_implemented"}))
    monkeypatch.setattr(settings, "inference_provider", provider)
    identity = await persisted_job(client, make_project, cap)
    async with async_session_factory() as session:
        result = await jobs_svc.execute_job(session, identity)
        assert result.status == JobStatus.failed and result.output is None
        assert "reported failure" in result.error
        assert await session.scalar(select(func.count()).select_from(Asset)) == 1
        await jobs_svc.execute_job(session, identity)
    assert len(calls) == 1
    assert resolve_availability(registry.get(cap)).executable


@pytest.mark.skipif(not TOOLS, reason="ffmpeg/ffprobe required")
@pytest.mark.parametrize(
    "body,mime,fmt",
    [
        (b"", "audio/mpeg", "mp3"),
        (b"", "audio/pcm", "pcm"),
        (b'{"status":"not_implemented"}', "application/json", "mp3"),
        (b'{"status":"not_implemented"}', "audio/mpeg", "mp3"),
        (b'{"status":"not_implemented"}', "audio/mpeg", "pcm"),
        (b'{"status":"not_implemented"}', "application/octet-stream", "pcm"),
        (b"not audio", "audio/mpeg", "mp3"),
        (b"x", "audio/pcm", "pcm"),
    ],
)
async def test_invalid_tts_never_publishes(client, make_project, monkeypatch, body, mime, fmt):
    calls = wire(monkeypatch, audio_response(body, mime))
    monkeypatch.setattr(settings, "inference_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_tts_response_format", fmt)
    identity = await persisted_job(client, make_project, "audio.tts")
    before = set(Path(settings.storage_dir).rglob("original/*"))
    async with async_session_factory() as session:
        result = await jobs_svc.execute_job(session, identity)
        assert result.status == JobStatus.failed and result.output is None
        assert await session.scalar(select(func.count()).select_from(Asset)) == 0
        await jobs_svc.execute_job(session, identity)
    assert len(calls) == 1
    assert set(Path(settings.storage_dir).rglob("original/*")) == before


@pytest.mark.skipif(not TOOLS, reason="ffmpeg/ffprobe required")
@pytest.mark.parametrize("representation", ["pcm", "mp3", "wav"])
async def test_valid_tts_publishes_real_audio(
    client, make_project, monkeypatch, tmp_path, representation
):
    audio = wav_bytes()
    mime = "audio/wav"
    if representation == "pcm":
        audio, mime = b"\x00\x00" * 2400, "audio/pcm"
    elif representation == "mp3":
        source, target = tmp_path / "source.wav", tmp_path / "speech.mp3"
        source.write_bytes(audio)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(source), str(target)], check=True, timeout=10
        )
        audio, mime = target.read_bytes(), "audio/mpeg"
    calls = wire(monkeypatch, audio_response(audio, mime))
    monkeypatch.setattr(settings, "inference_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_tts_response_format", representation)
    identity = await persisted_job(client, make_project, "audio.tts")
    async with async_session_factory() as session:
        result = await jobs_svc.execute_job(session, identity)
        assert result.status == JobStatus.succeeded
        descriptor = result.output["outputs"][0]
        asset = await session.get(Asset, uuid.UUID(descriptor["asset_id"]))
        assert asset.has_audio_stream and asset.duration_seconds > 0
        await jobs_svc.execute_job(session, identity)
        assert await session.scalar(select(func.count()).select_from(Asset)) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("cap", ["image.caption", "audio.transcribe", "audio.tts"])
async def test_cancellation_wins_during_invalid_provider_response(
    client, make_project, monkeypatch, cap
):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = wire(monkeypatch, json_response({"status": "not_implemented"}), gate=(entered, release))
    monkeypatch.setattr(
        settings, "inference_provider", "openrouter" if cap == "audio.tts" else "http"
    )
    identity = await persisted_job(client, make_project, cap)

    async def execute():
        async with async_session_factory() as session:
            return await jobs_svc.execute_job(session, identity)

    task = asyncio.create_task(execute())
    await asyncio.wait_for(entered.wait(), 5)
    async with async_session_factory() as session:
        await jobs_svc.cancel_job(session, identity)
    release.set()
    result = await task
    assert result.status == JobStatus.cancelled and result.output is None
    async with async_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Asset)) == (
            0 if cap == "audio.tts" else 1
        )
        await jobs_svc.execute_job(session, identity)
    assert len(calls) == 1


async def test_tts_malformed_internal_result_and_probe_failure_are_controlled(
    monkeypatch, tmp_path
):
    class Provider(BaseInferenceProvider):
        async def infer(self, request):
            return {AUDIO_BYTES_KEY: "not bytes"}

    monkeypatch.setattr("app.capabilities.handlers.audio_tts.get_provider", Provider)
    ctx = JobContext(capability_id="audio.tts", work_dir=tmp_path)
    with pytest.raises(ProviderError):
        await registry.get_handler("audio.tts").run(ctx)

    class ValidProvider(BaseInferenceProvider):
        async def infer(self, request):
            return {AUDIO_BYTES_KEY: wav_bytes(), AUDIO_FILENAME_KEY: "speech.wav"}

    async def failed_probe(*args):
        from app.services.media_metadata import MetadataProbeError

        raise MetadataProbeError("timeout")

    monkeypatch.setattr("app.capabilities.handlers.audio_tts.get_provider", ValidProvider)
    monkeypatch.setattr("app.capabilities.handlers.audio_tts.probe_media", failed_probe)
    with pytest.raises(ProviderError, match="unprobeable"):
        await registry.get_handler("audio.tts").run(ctx)


async def test_public_output_schemas_require_only_current_result_fields(client):
    for cap, field in [("image.caption", "caption"), ("audio.transcribe", "text")]:
        response = await client.get(f"/api/v1/capabilities/{cap}")
        assert response.status_code == 200
        assert response.json()["output_schema"]["required"] == [field]
        if field == "caption":
            assert response.json()["output_schema"]["properties"][field]["pattern"] == r"\S"
    response = await client.get("/api/v1/capabilities/audio.tts")
    assert "audio_url" in response.json()["output_schema"]["properties"]


@pytest.mark.skipif(not TOOLS, reason="ffmpeg/ffprobe required")
async def test_valid_tts_cancelled_after_probe_cannot_publish(client, make_project, monkeypatch):
    from app.capabilities.handlers import audio_tts

    calls = wire(monkeypatch, audio_response(wav_bytes(), "audio/wav"))
    monkeypatch.setattr(settings, "inference_provider", "openrouter")
    identity = await persisted_job(client, make_project, "audio.tts")
    entered, release = asyncio.Event(), asyncio.Event()
    probe = audio_tts.probe_media

    async def paused_probe(*args):
        metadata = await probe(*args)
        entered.set()
        await release.wait()
        return metadata

    monkeypatch.setattr(audio_tts, "probe_media", paused_probe)

    async def execute():
        async with async_session_factory() as session:
            return await jobs_svc.execute_job(session, identity)

    before = set(Path(settings.storage_dir).rglob("original/*"))
    task = asyncio.create_task(execute())
    await asyncio.wait_for(entered.wait(), 5)
    async with async_session_factory() as session:
        await jobs_svc.cancel_job(session, identity)
    release.set()
    result = await task
    assert result.status == JobStatus.cancelled and result.output is None
    async with async_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Asset)) == 0
        await jobs_svc.execute_job(session, identity)
    assert len(calls) == 1
    assert set(Path(settings.storage_dir).rglob("original/*")) == before
