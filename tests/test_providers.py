"""Unit tests for the OpenRouter inference provider (audio.tts)."""

from __future__ import annotations

import wave
from io import BytesIO

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import ProviderError
from app.providers.base import InferenceRequest
from app.providers.openrouter import (
    AUDIO_BYTES_KEY,
    AUDIO_FILENAME_KEY,
    OpenRouterInferenceProvider,
)


def _settings(**overrides) -> Settings:
    base = {
        "openrouter_api_key": "sk-test",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "openrouter_tts_model": "google/gemini-3.1-flash-tts-preview",
        "openrouter_tts_voice": "Kore",
        # Tests default to "mp3" (raw passthrough) unless exercising PCM wrapping.
        "openrouter_tts_response_format": "mp3",
        "openrouter_tts_pcm_sample_rate": 24000,
        "openrouter_site_url": "",
        "openrouter_site_name": "",
        "inference_timeout_seconds": 5.0,
    }
    base.update(overrides)
    return Settings(**base)


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes, headers: dict[str, str]) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers
        self.text = content.decode(errors="replace")


class _FakeAsyncClient:
    last_request: dict | None = None

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def post(self, url, headers=None, json=None):  # noqa: A002
        _FakeAsyncClient.last_request = {"url": url, "headers": headers, "json": json}
        return self._response


def _patch_client(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> None:
    monkeypatch.setattr(
        "app.providers.openrouter.httpx.AsyncClient",
        lambda timeout=None: _FakeAsyncClient(response),
    )


async def test_openrouter_tts_returns_audio_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_audio = b"\xff\xfb\x90\x00fake-mp3-bytes"
    _patch_client(
        monkeypatch,
        _FakeResponse(200, fake_audio, {"content-type": "audio/mpeg", "X-Generation-Id": "gen-1"}),
    )
    provider = OpenRouterInferenceProvider(_settings())

    result = await provider.infer(
        InferenceRequest(capability_id="audio.tts", payload={"text": "Hello there"})
    )

    assert result[AUDIO_BYTES_KEY] == fake_audio
    assert result[AUDIO_FILENAME_KEY] == "speech.mp3"
    assert result["provider"] == "openrouter"
    assert result["voice"] == "Kore"
    assert result["generation_id"] == "gen-1"
    sent = _FakeAsyncClient.last_request
    assert sent["url"] == "https://openrouter.ai/api/v1/audio/speech"
    assert sent["json"]["input"] == "Hello there"
    assert sent["json"]["response_format"] == "mp3"
    assert sent["headers"]["Authorization"] == "Bearer sk-test"


async def test_openrouter_tts_wraps_pcm_response_as_wav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_pcm = (b"\x01\x00\x02\x00") * 100
    _patch_client(
        monkeypatch, _FakeResponse(200, raw_pcm, {"content-type": "audio/pcm"})
    )
    provider = OpenRouterInferenceProvider(_settings(openrouter_tts_response_format="pcm"))

    result = await provider.infer(
        InferenceRequest(capability_id="audio.tts", payload={"text": "Hi"})
    )

    assert result[AUDIO_FILENAME_KEY] == "speech.wav"
    wav_bytes = result[AUDIO_BYTES_KEY]
    assert wav_bytes.startswith(b"RIFF")
    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 24000
        assert wav_file.readframes(wav_file.getnframes()) == raw_pcm
    assert _FakeAsyncClient.last_request["json"]["response_format"] == "pcm"


async def test_openrouter_tts_omits_site_headers_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch, _FakeResponse(200, b"audio", {"content-type": "audio/mpeg"}))
    provider = OpenRouterInferenceProvider(_settings())

    await provider.infer(InferenceRequest(capability_id="audio.tts", payload={"text": "Hi"}))

    headers = _FakeAsyncClient.last_request["headers"]
    assert "HTTP-Referer" not in headers
    assert "X-Title" not in headers


async def test_openrouter_tts_sends_site_headers_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch, _FakeResponse(200, b"audio", {"content-type": "audio/mpeg"}))
    provider = OpenRouterInferenceProvider(
        _settings(
            openrouter_site_url="https://editor.example.org",
            openrouter_site_name="UNICHE Media Editor",
        )
    )

    await provider.infer(InferenceRequest(capability_id="audio.tts", payload={"text": "Hi"}))

    headers = _FakeAsyncClient.last_request["headers"]
    assert headers["HTTP-Referer"] == "https://editor.example.org"
    assert headers["X-Title"] == "UNICHE Media Editor"


async def test_openrouter_tts_uses_requested_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, _FakeResponse(200, b"audio", {"content-type": "audio/mpeg"}))
    provider = OpenRouterInferenceProvider(_settings())

    result = await provider.infer(
        InferenceRequest(
            capability_id="audio.tts", payload={"text": "Hi", "voice": "verse"}
        )
    )

    assert result["voice"] == "verse"
    assert _FakeAsyncClient.last_request["json"]["voice"] == "verse"


async def test_openrouter_tts_requires_text(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenRouterInferenceProvider(_settings())
    with pytest.raises(ProviderError, match="non-empty 'text'"):
        await provider.infer(InferenceRequest(capability_id="audio.tts", payload={}))


async def test_openrouter_tts_requires_api_key() -> None:
    provider = OpenRouterInferenceProvider(_settings(openrouter_api_key=""))
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        await provider.infer(
            InferenceRequest(capability_id="audio.tts", payload={"text": "Hi"})
        )


async def test_openrouter_rejects_unsupported_capability() -> None:
    provider = OpenRouterInferenceProvider(_settings())
    with pytest.raises(ProviderError, match="does not support"):
        await provider.infer(InferenceRequest(capability_id="image.caption", payload={}))


async def test_openrouter_tts_error_status_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, _FakeResponse(402, b"insufficient credits", {}))
    provider = OpenRouterInferenceProvider(_settings())
    with pytest.raises(ProviderError, match="402"):
        await provider.infer(
            InferenceRequest(capability_id="audio.tts", payload={"text": "Hi"})
        )


async def test_openrouter_tts_rejects_non_audio_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(
        monkeypatch, _FakeResponse(200, b'{"error":"nope"}', {"content-type": "application/json"})
    )
    provider = OpenRouterInferenceProvider(_settings())
    with pytest.raises(ProviderError, match="unexpected content-type"):
        await provider.infer(
            InferenceRequest(capability_id="audio.tts", payload={"text": "Hi"})
        )


async def test_openrouter_tts_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TimeoutClient(_FakeAsyncClient):
        async def post(self, url, headers=None, json=None):  # noqa: A002
            raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(
        "app.providers.openrouter.httpx.AsyncClient",
        lambda timeout=None: _TimeoutClient(_FakeResponse(200, b"", {})),
    )
    provider = OpenRouterInferenceProvider(_settings())
    with pytest.raises(ProviderError, match="timed out"):
        await provider.infer(
            InferenceRequest(capability_id="audio.tts", payload={"text": "Hi"})
        )
