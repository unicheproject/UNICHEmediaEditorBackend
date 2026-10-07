"""Public ME-05 metadata, shared grounding and executable option contracts."""

import itertools
import subprocess
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.agent import catalog
from app.capabilities import registry
from app.capabilities.context import JobContext
from app.capabilities.handlers.audio_ops import AudioTranscodeHandler
from app.capabilities.validation import CapabilityInputValidationError, validate_capability_input
from app.core.errors import ValidationError
from tests.test_tool_capabilities import (
    API,
    HAVE_CONVERT,
    HAVE_FFMPEG,
    _ffprobe,
    _make_audio,
    _project,
    _run,
    _run_job,
    _upload,
)

AUDIO_OPTIONS = {
    "mp3": ["libmp3lame"],
    "wav": ["pcm_s16le"],
    "m4a": ["aac"],
    "flac": ["flac"],
    "ogg": ["libvorbis", "libopus"],
}
CODECS = ["libmp3lame", "pcm_s16le", "aac", "flac", "libvorbis", "libopus"]


async def test_colour_and_help_public_metadata(client: AsyncClient) -> None:
    caps = {c["id"]: c for c in (await client.get(f"{API}/capabilities")).json()}
    props = caps["image.colour.adjust"]["input_schema"]["properties"]
    for field in ("brightness", "saturation", "contrast"):
        prop = props[field]
        assert prop["type"] == "integer"
        assert prop["title"] == field.title()
        assert prop["x-unit"] == "percent"
        assert prop["default"] == (0 if field == "contrast" else 100)
        assert prop["minimum"] == (-100 if field == "contrast" else 0)
        assert "unchanged" in prop["description"]
    assert props["contrast"]["maximum"] == 100
    # Deliberate product bounds, not ImageMagick's technical limits.
    assert props["brightness"]["maximum"] == 200
    assert props["saturation"]["maximum"] == 200
    detail = (await client.get(f"{API}/capabilities/image.colour.adjust")).json()
    assert detail["input_schema"] == caps["image.colour.adjust"]["input_schema"]
    inpaint = caps["image.inpaint"]
    assert inpaint["availability"]["executable"] is False
    assert inpaint["readiness"]["implementation"] == "unavailable"
    help_props = inpaint["input_schema"]["properties"]
    assert help_props["mask_url"]["title"] == "Mask URL"
    assert "regions to fill or replace" in help_props["mask_url"]["description"]
    assert help_props["prompt"]["title"] == "Replacement description"
    assert "masked image regions" in help_props["prompt"]["description"]
    threshold = caps["video.shot.detect"]["input_schema"]["properties"]["threshold"]
    assert (threshold["default"], threshold["minimum"], threshold["maximum"]) == (27.0, 0, 255)
    assert "Lower values are more sensitive" in threshold["description"]
    assert "higher values detect fewer cuts" in threshold["description"]
    mix = caps["audio.mix"]["input_schema"]["properties"]["mode"]
    assert mix["x-enum-descriptions"] == {
        "mix": "Mix supplied audio with the primary asset's existing audio.",
        "duck": "Mix supplied audio at no more than a 0.2 volume multiplier.",
        "replace": "Replace the primary asset's audio with the supplied audio.",
    }


async def test_audio_transcode_public_choices_and_agent_grounding(client: AsyncClient) -> None:
    response = await client.get(f"{API}/capabilities/audio.transcode")
    assert response.status_code == 200
    schema = response.json()["input_schema"]
    props = schema["properties"]
    assert props["format"]["enum"] == list(AUDIO_OPTIONS)
    assert props["codec"]["enum"] == CODECS
    assert set(props["format"]["x-enum-descriptions"]) == set(AUDIO_OPTIONS)
    assert set(props["codec"]["x-enum-descriptions"]) == set(CODECS)
    assert props["codec"]["x-option-constraints"] == {
        "depends_on": "format", "allowed_values": AUDIO_OPTIONS,
    }
    assert schema["required"] == ["format"]
    assert "default" not in props["codec"]
    assert len(schema["allOf"]) == len(AUDIO_OPTIONS)
    agent = {c["capability_id"]: c for c in catalog.build_catalog()}
    assert agent["audio.transcode"]["input_schema"] == schema
    assert agent["image.colour.adjust"]["input_schema"] == registry.get(
        "image.colour.adjust"
    ).input_schema
    assert "image.inpaint" not in agent
    text = catalog.catalog_text()
    assert "choices_when_format=" in text
    assert "Uncompressed signed 16-bit PCM" in text


@pytest.mark.parametrize("fmt,codec", itertools.product(AUDIO_OPTIONS, CODECS))
def test_audio_transcode_all_schema_combinations(fmt: str, codec: str) -> None:
    cap = registry.get("audio.transcode")
    validate_capability_input(cap, {"format": fmt})
    if codec in AUDIO_OPTIONS[fmt]:
        validate_capability_input(cap, {"format": fmt, "codec": codec})
    else:
        with pytest.raises(CapabilityInputValidationError):
            validate_capability_input(cap, {"format": fmt, "codec": codec})


@pytest.mark.parametrize("params", [
    {"format": "xyz"}, {"format": ".mp3"}, {"format": "MP3"},
    {"format": "../mp3"}, {"format": "wav", "codec": "not-an-encoder"},
    {"format": "wav", "codec": "aac"}, {"format": "m4a", "codec": None},
])
async def test_unsupported_audio_choices_rejected_without_publication(
    client: AsyncClient, tmp_path: Path, params: dict,
) -> None:
    pid = await _project(client, "Unsupported audio choices")
    source = tmp_path / "source.wav"
    source.write_bytes(b"placeholder")
    aid = await _upload(client, pid, str(source), source.name, "audio/wav")
    response = await client.post(
        f"{API}/jobs", json={"capability_id": "audio.transcode", "asset_id": aid, "input": params}
    )
    assert response.status_code == 422
    assert (await client.get(f"{API}/projects/{pid}/jobs")).json()["total"] == 0
    assets = (await client.get(f"{API}/projects/{pid}/assets")).json()
    assert [a["id"] for a in assets] == [aid]
    # Direct/internal handler use observes the identical finite domain as admission.
    with pytest.raises(ValidationError):
        await AudioTranscodeHandler().run(JobContext(
            capability_id="audio.transcode", params=params,
            input_path=str(source), work_dir=tmp_path,
        ))


AUDIO_EXECUTIONS = [(fmt, None) for fmt in AUDIO_OPTIONS] + [
    (fmt, codec) for fmt, codecs in AUDIO_OPTIONS.items() for codec in codecs
]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize("fmt,codec", AUDIO_EXECUTIONS)
async def test_audio_transcode_real_supported_choices(
    client: AsyncClient, tmp_path: Path, fmt: str, codec: str | None,
) -> None:
    pid = await _project(client, "Audio options")
    source = tmp_path / "tone.wav"
    _make_audio(str(source))
    aid = await _upload(client, pid, str(source), source.name, "audio/wav")
    params = {"format": fmt}
    if codec is not None:
        params["codec"] = codec
    job = await _run_job(client, "audio.transcode", aid, params)
    assert job["status"] == "succeeded", job
    assert len(job["output"]["outputs"]) == 1
    output = job["output"]["outputs"][0]
    assert output["filename"].endswith(f".{fmt}")
    assert output["media_type"] == "audio"
    rendered = tmp_path / f"result.{fmt}"
    rendered.write_bytes((await client.get(output["download_path"])).content)
    probe = _ffprobe(str(rendered))
    stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    expected = {
        "libmp3lame": "mp3", "pcm_s16le": "pcm_s16le", "aac": "aac",
        "flac": "flac", "libvorbis": "vorbis", "libopus": "opus",
    }
    assert stream["codec_name"] == expected[codec or AUDIO_OPTIONS[fmt][0]]
    assert {"mp3": "mp3", "wav": "wav", "m4a": "mov", "flac": "flac", "ogg": "ogg"}[
        fmt
    ] in probe["format"]["format_name"].split(",")
    assert float(probe["format"]["duration"]) == pytest.approx(2, abs=0.1)
    # Successful probing alone is not proof of playable compressed audio.
    _run(["ffmpeg", "-v", "error", "-i", str(rendered), "-f", "null", "-"])
    asset = (await client.get(f"{API}/assets/{output['asset_id']}")).json()
    assert asset["technical_metadata_status"] == "available"
    assert asset["audio_codec"] == stream["codec_name"]


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
@pytest.mark.parametrize("params", [
    {}, {"brightness": 0}, {"brightness": 200}, {"saturation": 0},
    {"saturation": 200}, {"brightness": 150}, {"saturation": 150},
    {"contrast": -100}, {"contrast": 100},
])
async def test_colour_adjust_real_boundaries(
    client: AsyncClient, tmp_path: Path, params: dict,
) -> None:
    pid = await _project(client, "Colour boundaries")
    source = tmp_path / "colour.png"
    _run(["convert", "-size", "16x16", "xc:#806040", str(source)])
    aid = await _upload(client, pid, str(source), source.name, "image/png")
    job = await _run_job(client, "image.colour.adjust", aid, params)
    assert job["status"] == "succeeded", job
    output = job["output"]["outputs"][0]
    rendered = tmp_path / "result.png"
    rendered.write_bytes((await client.get(output["download_path"])).content)
    data = subprocess.check_output(["convert", str(rendered), "-depth", "8", "RGB:-"])
    assert len(data) == 16 * 16 * 3
    if params.get("brightness") == 0:
        assert not any(data)
    if params.get("saturation") == 0:
        assert data[0] == data[1] == data[2]
    if not params:
        assert data[:3] == bytes([128, 96, 64])


@pytest.mark.parametrize("params", [
    {"brightness": -1}, {"saturation": -1}, {"contrast": -101}, {"contrast": 101},
    {"brightness": 201}, {"saturation": 201}, {"brightness": 300}, {"saturation": 300},
])
def test_colour_adjust_invalid_boundaries(params: dict) -> None:
    with pytest.raises(CapabilityInputValidationError):
        validate_capability_input(registry.get("image.colour.adjust"), params)


@pytest.mark.parametrize("field,value", itertools.product(
    ("brightness", "saturation"), (-1, 201, 300),
))
async def test_colour_product_range_rejected_before_job_creation(
    client: AsyncClient, tmp_path: Path, field: str, value: int,
) -> None:
    pid = await _project(client, "Colour product range")
    source = tmp_path / "colour.png"
    source.write_bytes(b"placeholder")
    aid = await _upload(client, pid, str(source), source.name, "image/png")
    response = await client.post(
        f"{API}/jobs",
        json={"capability_id": "image.colour.adjust", "asset_id": aid, "input": {field: value}},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert (await client.get(f"{API}/projects/{pid}/jobs")).json()["total"] == 0
    assert [a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()] == [aid]
