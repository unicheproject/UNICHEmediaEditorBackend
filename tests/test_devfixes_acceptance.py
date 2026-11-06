"""Regression checks named after the D5.1 "devfixes - Image tools" report items.

Each test reproduces the reported scenario against the backend contract. Items
fixed in their own modules have detailed tests there; these are the report's
own cases, kept together so the report can be re-checked in one run.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.capabilities.definitions import CAPABILITIES
from app.core.database import async_session_factory
from app.services.media_metadata import probe_asset_metadata

API = "/api/v1"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_CONVERT = shutil.which("convert") is not None
IMPLEMENTATION_NAMES = re.compile(
    r"FFmpeg|ImageMagick|Real-ESRGAN|ESRGAN|RNNoise|Whisper|loudnorm|libx26|PySceneDetect",
    re.IGNORECASE,
)


def _walk_enums(schema: dict, name: str | None = None):
    if "enum" in schema and name:
        yield name, schema
    for key, sub in schema.get("properties", {}).items():
        yield from _walk_enums(sub, key)
    if isinstance(schema.get("items"), dict):
        yield from _walk_enums(schema["items"], name)


@pytest.mark.parametrize("capability", CAPABILITIES, ids=lambda c: c.id)
def test_u6_every_choice_is_explained(capability) -> None:
    for name, schema in _walk_enums(capability.input_schema):
        described = schema.get("x-enum-descriptions") or {}
        assert all(str(value) in described for value in schema["enum"]), (
            f"{capability.id}.{name} has choices without x-enum-descriptions"
        )


@pytest.mark.parametrize("capability", CAPABILITIES, ids=lambda c: c.id)
def test_u7_user_text_does_not_name_the_implementation(capability) -> None:
    texts = [capability.summary or ""]
    texts += [
        prop.get("x-help", "") for prop in capability.input_schema.get("properties", {}).values()
    ]
    for text in texts:
        assert not IMPLEMENTATION_NAMES.search(text), (capability.id, text)


@pytest.mark.parametrize("gain_db", [-61, 31])
def test_c9_gain_has_a_bounded_range(gain_db: float) -> None:
    from app.capabilities import registry
    from app.capabilities.validation import (
        CapabilityInputValidationError,
        validate_capability_input,
    )

    cap = registry.get("audio.gain")
    validate_capability_input(cap, {"gain_db": 0})
    validate_capability_input(cap, {"gain_db": -60})
    with pytest.raises(CapabilityInputValidationError):
        validate_capability_input(cap, {"gain_db": gain_db})


def test_u5_resize_help_does_not_describe_encoder_tricks() -> None:
    for capability in CAPABILITIES:
        if capability.id in {"video.resize", "image.resize"}:
            for side in ("width", "height"):
                prop = capability.input_schema["properties"][side]
                text = f"{prop['description']} {prop.get('x-help', '')}"
                assert "-1" not in text and "FFmpeg" not in text, text
                assert prop["minimum"] >= 1



def _all_text(capability_id: str) -> str:
    """Every description a client or the Agent can show for a capability."""
    cap = next(c for c in CAPABILITIES if c.id == capability_id)
    parts = [cap.description, cap.summary or ""]
    for prop in cap.input_schema.get("properties", {}).values():
        parts += [prop.get("description", ""), prop.get("x-help", "")]
        parts += list(prop.get("x-enum-descriptions", {}).values())
    return " ".join(parts)


def test_c4_compressed_does_not_promise_a_smaller_file() -> None:
    # The cap covers the video bitrate only, and only when it is known.
    text = _all_text("video.transcode")
    assert "compressed always" not in text.lower()
    assert "kept below the source's bitrate" not in text
    assert "not guaranteed" in text and "not always" in text


def test_c6_video_soundtrack_codec_is_not_promised() -> None:
    # A video's soundtrack is re-encoded for the container (AAC or Opus).
    for capability_id in ("audio.gain", "audio.normalize", "audio.fade", "audio.denoise"):
        text = _all_text(capability_id)
        assert "keeps its source sample rate, channels, codec and bitrate" not in text
        assert "re-encoded as AAC" in text, capability_id


def test_d08_social_export_says_it_enlarges() -> None:
    # Social fits and pads to 1080 px, so smaller sources are scaled up.
    for capability_id in ("video.export", "project.export"):
        text = _all_text(capability_id)
        assert "and Preview are quality-targeted H.264/AAC MP4 that never upscale" not in text
        assert "enlarg" in text, capability_id


async def _upload(client: AsyncClient, pid: str, path: Path, mime: str) -> str:
    resp = await client.post(
        f"{API}/projects/{pid}/assets", files={"file": (path.name, path.read_bytes(), mime)}
    )
    assert resp.status_code == 201, resp.text
    asset_id: str = resp.json()["id"]
    async with async_session_factory() as session:
        await probe_asset_metadata(session, uuid.UUID(asset_id))
    return asset_id


async def _submit(client: AsyncClient, capability: str, asset_id: str, params: dict):
    return await client.post(f"{API}/jobs", json={
        "capability_id": capability, "asset_id": asset_id, "input": params,
    })


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
async def test_c1_c3_image_resize_and_crop_report_cases(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "test_image.jpg"
    subprocess.run(["convert", "-size", "2400x1600", "xc:gray", str(src)], check=True)
    pid = await make_project("devfixes-image")
    aid = await _upload(client, pid, src, "image/jpeg")

    # C-3: the job the dialog actually sent; negative offset and too tall.
    resp = await _submit(client, "image.crop", aid,
                         {"x": 242, "y": -154, "width": 1919, "height": 1908})
    assert resp.status_code == 422
    resp = await _submit(client, "image.crop", aid,
                         {"x": 242, "y": 0, "width": 1919, "height": 1908})
    assert resp.status_code == 422
    assert "2400×1600" in resp.json()["error"]["message"]

    # C-1: one side keeps the aspect; forcing both reports the distortion.
    job = (await client.get(f"{API}/jobs/{(await _submit(
        client, 'image.resize', aid, {'width': 1600})).json()['id']}")).json()
    assert (job["output"]["width"], job["output"]["height"]) == (1600, 1067)
    job = (await client.get(f"{API}/jobs/{(await _submit(
        client, 'image.resize', aid, {'width': 1600, 'height': 1000})).json()['id']}")).json()
    assert job["output"]["aspect_ratio_change"] > 0.06
    # C-5: every file job compares its output with its input.
    assert job["output"]["size_comparison"]["input_size_bytes"] == src.stat().st_size


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
async def test_c9_c10_trim_report_cases(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "video_main.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=duration=3:size=320x240:rate=25", "-pix_fmt", "yuv420p",
                    str(src)], check=True)
    pid = await make_project("devfixes-trim")
    aid = await _upload(client, pid, src, "video/mp4")
    # C-9: the dialog's default end of 00:00.001 is a degenerate range.
    assert (await _submit(client, "video.trim", aid, {"start": 0, "end": 0.001})).status_code \
        == 422
    assert (await _submit(client, "video.trim", aid, {"start": 2, "end": 1})).status_code == 422
    # C-10: keep from a time onward without knowing the duration.
    resp = await _submit(client, "video.trim", aid, {"start": 1})
    assert resp.status_code == 201, resp.text
    job = (await client.get(f"{API}/jobs/{resp.json()['id']}")).json()
    assert job["status"] == "succeeded"
    assert job["output"]["outputs"][0]["filename"] == "video_main_trim_1-end.mp4"
