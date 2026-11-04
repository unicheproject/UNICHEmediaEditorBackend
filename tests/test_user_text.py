"""User-facing capability text (summary / x-help), separate from the Agent contract."""

from __future__ import annotations

import re

import pytest
from httpx import AsyncClient

from app.agent.catalog import catalog_text
from app.capabilities import registry
from app.capabilities.definitions import CAPABILITIES_BY_ID
from app.capabilities.user_text import USER_TEXT, apply_user_text

API = "/api/v1"

# Groups completed so far; every capability in them needs a summary and every
# parameter an x-help.
COVERED_GROUPS = {
    "video editing": {
        "video.trim", "video.split", "video.concat", "video.transcode", "video.mute",
        "video.crop", "video.resize", "video.thumbnail", "video.audio.extract",
    },
    "image editing": {
        "image.resize", "image.crop", "image.format", "image.colour.adjust",
        "image.upscale",
    },
    "audio": {
        "audio.trim", "audio.concat", "audio.gain", "audio.normalize", "audio.fade",
        "audio.transcode", "audio.denoise", "audio.mix",
    },
    "composition and export": {
        "image.slideshow", "image.kenburns", "media.titlecard", "video.lowerthird",
        "video.subtitle.embed", "video.compose", "project.export",
    },
}
# Implementation vocabulary the target user does not have.
JARGON = (
    "ffmpeg", "imagemagick", "rnnoise", "pyscenedetect", "real-esrgan", "libx264",
    "libx265", "libvpx", "codec", "encoder", "container", "crf", "lufs",
)
# FFmpeg's "-1 means automatic" sentinel, but not ordinary values such as -16.
SENTINEL = re.compile(r"(?<![\d.])-1(?![\d.])")


def _texts(capability_id: str) -> list[str]:
    cap = registry.get(capability_id)
    params = cap.input_schema.get("properties", {}).values()
    return [cap.summary or ""] + [p["x-help"] for p in params if "x-help" in p]


@pytest.mark.parametrize("group", sorted(COVERED_GROUPS))
def test_covered_groups_are_complete(group: str) -> None:
    for capability_id in COVERED_GROUPS[group]:
        cap = registry.get(capability_id)
        assert cap.summary, capability_id
        for name, prop in cap.input_schema.get("properties", {}).items():
            assert prop.get("x-help"), f"{capability_id}.{name}"


@pytest.mark.parametrize("capability_id", sorted(USER_TEXT))
def test_user_text_avoids_implementation_vocabulary(capability_id: str) -> None:
    for text in _texts(capability_id):
        lowered = text.lower()
        assert not [word for word in JARGON if word in lowered], (capability_id, text)
        assert not SENTINEL.search(text), (capability_id, text)
        assert not [cid for cid in CAPABILITIES_BY_ID if cid in text], (capability_id, text)


def test_agent_prompt_excludes_user_text_and_keeps_descriptions() -> None:
    prompt = catalog_text()
    for capability_id, entry in USER_TEXT.items():
        cap = registry.get(capability_id)
        if entry.get("summary"):
            assert entry["summary"] not in prompt
        for help_text in entry.get("params", {}).values():
            assert help_text not in prompt
        if cap.enabled and f"- {capability_id} " in prompt:
            assert cap.description in prompt


def test_user_text_drift_fails_loudly() -> None:
    caps = [registry.get("video.mute")]
    with pytest.raises(RuntimeError, match="unknown capabilities"):
        apply_user_text(caps, {"video.nope": {"summary": "x"}})
    with pytest.raises(RuntimeError, match="unknown parameter"):
        apply_user_text(caps, {"video.mute": {"params": {"missing": "x"}}})


def test_capabilities_without_text_report_null_summary() -> None:
    uncovered = [cid for cid in CAPABILITIES_BY_ID if cid not in USER_TEXT]
    assert uncovered
    assert all(CAPABILITIES_BY_ID[cid].summary is None for cid in uncovered)


async def test_api_exposes_summary_and_x_help(client: AsyncClient) -> None:
    listed = {c["id"]: c for c in (await client.get(f"{API}/capabilities")).json()}
    assert listed["video.trim"]["summary"] == USER_TEXT["video.trim"]["summary"]
    end = listed["video.trim"]["input_schema"]["properties"]["end"]
    assert end["x-help"] == USER_TEXT["video.trim"]["params"]["end"]
    assert end["description"] != end["x-help"]
    detail = (await client.get(f"{API}/capabilities/video.resize")).json()
    assert detail["summary"] == USER_TEXT["video.resize"]["summary"]
