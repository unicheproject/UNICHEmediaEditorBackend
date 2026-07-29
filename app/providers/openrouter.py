"""OpenRouter inference provider.

Backs `audio.tts` (via `/audio/speech`) and `audio.transcribe` (via
`/audio/transcriptions`) -- see https://openrouter.ai/docs/guides/overview/multimodal/stt.

`audio.tts`: returns the synthesized audio as raw bytes under a private key
that `AudioTtsHandler` lifts out and writes to a file. `response_format` is
model-dependent: some models only accept "pcm" (e.g. the default
`google/gemini-3.1-flash-tts-preview`) and error on "mp3", others only accept
"mp3". "pcm" comes back as headerless raw audio, which isn't a playable file
on its own, so it's wrapped in a WAV container here before being handed to the
handler.

Voice names are also model-specific and non-obvious: the Gemini model 500s on
OpenAI-style names like "alloy"/"verse" and needs one of its own native voices
(e.g. "Kore", "Puck") — see openrouter_tts_voice in app/core/config.py.

`audio.transcribe`: uploads the input asset as multipart/form-data (the
OpenAI-compatible form, not the JSON+base64 alternative -- avoids base64
inflating a potentially large audio file) to a real Whisper-class model,
requesting `verbose_json` so the response includes the detected language
alongside the transcript text.
"""

from __future__ import annotations

import mimetypes
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx

from app.core.config import Settings
from app.core.errors import ProviderError
from app.providers.base import BaseInferenceProvider, InferenceRequest

AUDIO_BYTES_KEY = "_audio_bytes"
AUDIO_FILENAME_KEY = "_audio_filename"


class OpenRouterInferenceProvider(BaseInferenceProvider):
    name = "openrouter"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def infer(self, request: InferenceRequest) -> dict[str, Any]:
        if request.capability_id == "audio.tts":
            return await self._speech(request)
        if request.capability_id == "audio.transcribe":
            return await self._transcribe(request)
        raise ProviderError(
            f"OpenRouter provider does not support capability "
            f"'{request.capability_id}'"
        )

    async def _speech(self, request: InferenceRequest) -> dict[str, Any]:
        api_key = self._settings.openrouter_api_key
        if not api_key:
            raise ProviderError("OPENROUTER_API_KEY is not configured")

        text = request.payload.get("text")
        if not text:
            raise ProviderError("audio.tts requires a non-empty 'text' param")
        voice = request.payload.get("voice") or self._settings.openrouter_tts_voice
        response_format = (
            request.payload.get("response_format")
            or self._settings.openrouter_tts_response_format
        )

        url = f"{self._settings.openrouter_base_url.rstrip('/')}/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # Optional: attributes the request for OpenRouter's public rankings.
        if self._settings.openrouter_site_url:
            headers["HTTP-Referer"] = self._settings.openrouter_site_url
        if self._settings.openrouter_site_name:
            headers["X-Title"] = self._settings.openrouter_site_name
        body = {
            "model": self._settings.openrouter_tts_model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._settings.inference_timeout_seconds
            ) as client:
                response = await client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"OpenRouter TTS request timed out: {url}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenRouter TTS request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"OpenRouter TTS endpoint returned {response.status_code}: "
                f"{response.text[:500]}"
            )

        content_type = response.headers.get("content-type", "")
        if "audio" not in content_type and "octet-stream" not in content_type:
            raise ProviderError(
                f"OpenRouter TTS endpoint returned unexpected content-type "
                f"'{content_type}'"
            )

        audio_bytes = response.content
        filename = "speech.mp3"
        if response_format == "pcm" or "pcm" in content_type:
            audio_bytes = _wrap_pcm_as_wav(
                audio_bytes, sample_rate=self._settings.openrouter_tts_pcm_sample_rate
            )
            filename = "speech.wav"

        return {
            AUDIO_BYTES_KEY: audio_bytes,
            AUDIO_FILENAME_KEY: filename,
            "voice": voice,
            "model": self._settings.openrouter_tts_model,
            "response_format": response_format,
            "provider": self.name,
            "generation_id": response.headers.get("X-Generation-Id"),
        }

    async def _transcribe(self, request: InferenceRequest) -> dict[str, Any]:
        api_key = self._settings.openrouter_api_key
        if not api_key:
            raise ProviderError("OPENROUTER_API_KEY is not configured")
        if not request.file_path:
            raise ProviderError("audio.transcribe requires an input asset")

        path = Path(request.file_path)
        audio_bytes = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        model = request.payload.get("model") or self._settings.openrouter_transcribe_model

        url = f"{self._settings.openrouter_base_url.rstrip('/')}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}
        if self._settings.openrouter_site_url:
            headers["HTTP-Referer"] = self._settings.openrouter_site_url
        if self._settings.openrouter_site_name:
            headers["X-Title"] = self._settings.openrouter_site_name

        form_data = {"model": model, "response_format": "verbose_json"}
        language = request.payload.get("language")
        if language:
            form_data["language"] = language

        try:
            async with httpx.AsyncClient(
                timeout=self._settings.inference_timeout_seconds
            ) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    data=form_data,
                    files={"file": (path.name, audio_bytes, content_type)},
                )
        except httpx.TimeoutException as exc:
            raise ProviderError(f"OpenRouter transcription request timed out: {url}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenRouter transcription request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"OpenRouter transcription endpoint returned {response.status_code}: "
                f"{response.text[:500]}"
            )

        body = response.json()
        return {
            "text": body.get("text", ""),
            "language": body.get("language") or language,
            "model": model,
            "provider": self.name,
            "generation_id": response.headers.get("X-Generation-Id"),
        }


def _wrap_pcm_as_wav(
    pcm_bytes: bytes, *, sample_rate: int, channels: int = 1, sample_width: int = 2
) -> bytes:
    """Wrap headerless 16-bit PCM in a WAV container so it's a playable file.

    Sample rate/channels aren't in OpenRouter's /audio/speech docs, but the
    response's own content-type header confirms them per-request (e.g.
    "audio/pcm;rate=24000;channels=1"), matching the 24kHz mono default here.
    """
    buf = BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buf.getvalue()
