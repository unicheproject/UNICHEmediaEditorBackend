"""MED/041: audio.transcribe takes the same optional language hint as subtitles (offline)."""

from __future__ import annotations

import io
import json

import pytest
from httpx import AsyncClient

from app.capabilities import registry
from app.capabilities.validation import (
    CapabilityInputValidationError,
    validate_capability_input,
)
from app.core.config import settings
from tests.test_provider_semantics import json_response, wire

API = "/api/v1"
BAD_LANGUAGES = ["english", "EL", "e", "ell", "", 5, None]


def test_language_is_optional_and_shares_the_subtitle_rule() -> None:
    transcribe = registry.get("audio.transcribe").input_schema
    subtitles = registry.get("subtitle.autogenerate").input_schema
    assert "language" not in transcribe.get("required", [])
    assert transcribe["properties"]["language"]["pattern"] == "^[a-z]{2}$"
    assert (transcribe["properties"]["language"]["pattern"]
            == subtitles["properties"]["language"]["pattern"])
    for params in ({}, {"language": "el"}, {"language": "en"}):
        validate_capability_input(registry.get("audio.transcribe"), params)


@pytest.mark.parametrize("language", BAD_LANGUAGES)
def test_invalid_language_is_rejected_for_jobs_and_agent_plans(language: object) -> None:
    for allow_step_refs in (False, True):
        with pytest.raises(CapabilityInputValidationError, match="input.language"):
            validate_capability_input(
                registry.get("audio.transcribe"), {"language": language},
                allow_step_refs=allow_step_refs,
            )


async def _audio_asset(client: AsyncClient, make_project, name: str) -> tuple[str, str]:
    pid = await make_project(name)
    aid = (await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": ("clip.mp3", io.BytesIO(b"bytes"), "audio/mpeg")},
    )).json()["id"]
    return pid, aid


async def _transcribe(client: AsyncClient, aid: str, params: dict) -> dict:
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "audio.transcribe", "asset_id": aid, "input": params,
    })
    assert resp.status_code == 201, resp.text
    job: dict = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    return job


@pytest.mark.parametrize("language", ["english", "EL"])
async def test_invalid_language_returns_422_before_job_creation(
    client: AsyncClient, make_project, language: str
) -> None:
    pid, aid = await _audio_asset(client, make_project, f"transcribe-bad-{language}")
    resp = await client.post(f"{API}/jobs", json={
        "capability_id": "audio.transcribe", "asset_id": aid, "input": {"language": language},
    })
    assert resp.status_code == 422
    assert "input.language" in resp.json()["error"]["message"]
    assert (await client.get(f"{API}/projects/{pid}/jobs")).json()["total"] == 0


@pytest.mark.parametrize("language", ["el", None])
async def test_openrouter_sends_the_hint_only_when_given(
    client: AsyncClient, make_project, monkeypatch, language: str | None
) -> None:
    calls = wire(monkeypatch, json_response({"text": "Καλημέρα", "language": "greek"}))
    monkeypatch.setattr(settings, "inference_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-test")
    _, aid = await _audio_asset(client, make_project, f"transcribe-or-{language}")

    job = await _transcribe(client, aid, {"language": language} if language else {})
    assert job["status"] == "succeeded", job
    assert job["output"]["text"] == "Καλημέρα" and job["output"]["language"] == "greek"
    assert "segments" not in job["output"]  # transcription output is unchanged
    body = calls[0].content
    assert (b'name="language"' in body) is (language is not None)
    if language:
        assert b'name="language"\r\n\r\nel\r\n' in body


@pytest.mark.parametrize("language", ["el", None])
async def test_http_provider_receives_the_hint_in_the_job_input(
    client: AsyncClient, make_project, monkeypatch, language: str | None
) -> None:
    calls = wire(monkeypatch, json_response({"text": "Hello", "language": "en"}))
    monkeypatch.setattr(settings, "inference_provider", "http")
    _, aid = await _audio_asset(client, make_project, f"transcribe-http-{language}")

    job = await _transcribe(client, aid, {"language": language} if language else {})
    assert job["status"] == "succeeded", job
    body = calls[0].content.decode("utf-8", "replace")
    marker = 'name="input"\r\n\r\n'
    sent = json.loads(body[body.index(marker) + len(marker):].split("\r\n", 1)[0])
    assert sent.get("language") == language
