"""Planner backends (mock + OpenRouter) and the propose/validate/repair loop.

- MockPlanner: deterministic, rule-based; works offline and in tests.
- OpenRouterBackend: real LLM planning via OpenRouter, preserving
  `reasoning_details` across turns so clarifications continue the same reasoning.

`propose()` runs the backend, parses the result, validates a Plan against the
registry, and (for LLM backends) feeds validation errors back for a bounded
repair retry.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.agent import catalog as catalog_mod
from app.agent.planner import PlanValidationError, validate_plan
from app.agent.schemas import Clarification, Plan, PlannerResult
from app.core.config import Settings
from app.core.config import settings as default_settings
from app.core.errors import ProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_INSTRUCTIONS = """\
You are a media-editing planner. Given a user request and a catalog of
capabilities, output ONLY JSON: either
  {"type":"clarification","question":"...","missing":["..."]}
when required parameters cannot be inferred, or
  {"type":"plan","summary":"...","steps":[
     {"id":"step1","capability_id":"...","params":{...},
      "asset":"<asset_id or @stepN>","assets":["<id or @stepN>"...],
      "depends_on":["stepN"]}]}.
Rules: use only capability_ids from the catalog. `asset` is the primary input
(an uploaded asset id, or "@stepN" for a prior step's output); `assets` is the
ordered list for multi-input ops (slideshow/concat/compose). Put all other
inputs (width, height, start, end, format, subtitle_asset_id, music_asset_id,
audio_asset_id, ...) in `params`. Map "1080p"->1920x1080, "720p"->1280x720,
"square"->1080x1080; timecodes (00:01:10) -> seconds. Only plan steps whose
capabilities exist in the catalog.
"""


# --------------------------------------------------------------------------
# OpenRouter backend
# --------------------------------------------------------------------------
class OpenRouterBackend:
    name = "openrouter"

    def __init__(self, settings: Settings) -> None:
        self._s = settings

    async def generate(self, messages: list[dict]) -> dict:
        if not self._s.openrouter_api_key:
            raise ProviderError("OPENROUTER_API_KEY is not configured")
        url = f"{self._s.openrouter_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self._s.openrouter_model,
            "messages": messages,
            "reasoning": {"enabled": True},
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=self._s.agent_timeout_seconds) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {self._s.openrouter_api_key}"},
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenRouter request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise ProviderError(f"OpenRouter returned {resp.status_code}: {resp.text[:300]}")
        # Return the raw assistant message so reasoning_details can be preserved.
        return resp.json()["choices"][0]["message"]


# --------------------------------------------------------------------------
# Mock backend (deterministic, offline)
# --------------------------------------------------------------------------
_DIMENSIONS = {"1080p": (1920, 1080), "720p": (1280, 720), "square": (1080, 1080)}


def _dims(text: str) -> tuple[int, int]:
    for key, wh in _DIMENSIONS.items():
        if key in text:
            return wh
    return (1920, 1080)


class MockPlanner:
    name = "mock"

    def propose(self, user_text: str, assets_by_id: dict[str, str]) -> dict:
        t = user_text.lower()
        images = [a for a, m in assets_by_id.items() if m == "image"]
        videos = [a for a, m in assets_by_id.items() if m == "video"]
        audios = [a for a, m in assets_by_id.items() if m == "audio"]
        w, h = _dims(t)

        if "slideshow" in t:
            steps: list[dict] = []
            compose_inputs: list[str] = []
            m = re.search(r"title card[^\"']*[\"']([^\"']+)[\"']", t) or re.search(
                r"title[^\"']*[\"']([^\"']+)[\"']", user_text
            )
            if m or "title card" in t:
                title = m.group(1) if m else "Title"
                steps.append({
                    "id": "title", "capability_id": "media.titlecard",
                    "params": {"text": title, "duration": 3, "width": w, "height": h},
                })
                compose_inputs.append("@title")
            spi = re.search(r"(\d+)\s*second", t)
            steps.append({
                "id": "slides", "capability_id": "image.slideshow",
                "params": {"asset_ids": images,
                           "seconds_per_image": int(spi.group(1)) if spi else 4,
                           "width": w, "height": h},
                "assets": images,
            })
            compose_inputs.append("@slides")
            params: dict[str, Any] = {"asset_ids": compose_inputs, "width": w, "height": h}
            if audios:
                params["audio_asset_id"] = audios[0]
                params["audio_mode"] = "replace"
            steps.append({
                "id": "export", "capability_id": "video.compose",
                "params": params, "assets": compose_inputs,
                "depends_on": [s["id"] for s in steps],
            })
            return {"type": "plan", "summary": "Slideshow with title card", "steps": steps}

        if ("trim" in t or "cut" in t) and not re.search(r"\d", t):
            return {"type": "clarification",
                    "question": "What time range should I trim to (start and end, in seconds)?",
                    "missing": ["start", "end"]}

        if "thumbnail" in t and videos:
            ts = re.search(r"(\d+)", t)
            return {"type": "plan", "summary": "Extract thumbnail", "steps": [{
                "id": "thumb", "capability_id": "video.thumbnail",
                "params": {"timestamp": int(ts.group(1)) if ts else 1}, "asset": videos[0]}]}

        if "mute" in t and videos:
            return {"type": "plan", "summary": "Mute video", "steps": [{
                "id": "mute", "capability_id": "video.mute", "params": {}, "asset": videos[0]}]}

        if "resize" in t and videos:
            return {"type": "plan", "summary": "Resize video", "steps": [{
                "id": "resize", "capability_id": "video.resize",
                "params": {"width": w, "height": h}, "asset": videos[0]}]}

        return {"type": "clarification",
                "question": "I couldn't map that to the available tools. Could you rephrase "
                            "in terms of trim/crop/resize/convert/slideshow/title card/compose?",
                "missing": []}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _parse(raw: dict | str) -> PlannerResult:
    data = json.loads(raw) if isinstance(raw, str) else raw
    if data.get("type") == "plan":
        return Plan.model_validate(data)
    return Clarification.model_validate(data)


def _system_message() -> dict:
    return {"role": "system", "content": SYSTEM_INSTRUCTIONS + "\n\nCAPABILITIES:\n"
            + catalog_mod.catalog_text()}


async def propose(
    history: list[dict],
    user_message: str,
    assets_by_id: dict[str, str],
    *,
    settings: Settings | None = None,
) -> tuple[PlannerResult, list[dict]]:
    """Return (result, new_messages_to_append) for the session transcript."""
    settings = settings or default_settings
    new_messages: list[dict] = [{"role": "user", "content": user_message}]

    if settings.agent_provider == "mock":
        raw = MockPlanner().propose(user_message, assets_by_id)
        result = _parse(raw)
        new_messages.append({"role": "assistant", "content": json.dumps(raw)})
        if isinstance(result, Plan):
            # surfaces as error if the mock is wrong
            validate_plan(
                result, assets_by_id, deterministic_only=settings.agent_deterministic_only
            )
        return result, new_messages

    backend = OpenRouterBackend(settings)
    asset_ctx = {"role": "system", "content": "IN-SCOPE ASSETS (id: media_type): "
                 + json.dumps(assets_by_id)}
    messages = [_system_message(), asset_ctx, *history, {"role": "user", "content": user_message}]

    last_error: str | None = None
    for attempt in range(settings.agent_max_repair_retries + 1):
        if last_error:
            messages.append({
                "role": "user",
                "content": f"That plan was invalid: {last_error}. Fix and resend JSON only.",
            })
        assistant = await backend.generate(messages)
        # Preserve reasoning_details unmodified across turns.
        stored = {"role": "assistant", "content": assistant.get("content")}
        if assistant.get("reasoning_details") is not None:
            stored["reasoning_details"] = assistant["reasoning_details"]
        messages.append(assistant)
        if attempt == 0:
            new_messages.append(stored)
        else:
            new_messages[-1] = stored

        try:
            result = _parse(assistant.get("content") or "{}")
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"output was not valid JSON ({exc})"
            continue
        if isinstance(result, Plan):
            try:
                validate_plan(
                    result, assets_by_id,
                    deterministic_only=settings.agent_deterministic_only,
                )
            except PlanValidationError as exc:
                last_error = str(exc)
                continue
        return result, new_messages

    raise ProviderError(f"Planner could not produce a valid plan: {last_error}")
