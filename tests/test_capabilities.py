import re

from httpx import AsyncClient

from app.capabilities import registry
from app.capabilities.validation import validate_capability_input
from app.schemas.capability import (
    CapabilityLayer,
    PrimaryAssetApplicability,
    ProjectContextApplicability,
)

API = "/api/v1"

_INTERNAL_PRESENTATION_REFERENCE = re.compile(
    r"\b(?:input|params|ctx)\.[A-Za-z_][A-Za-z0-9_.]*"
    r"|\b[A-Za-z][A-Za-z0-9]*_asset_ids?\b"
)


def _presentation_text(value: object, path: str = "") -> list[tuple[str, str]]:
    """Collect only human-facing titles, descriptions and enum explanations."""
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key in ("title", "description"):
            text = value.get(key)
            if isinstance(text, str):
                found.append((f"{path}.{key}", text))
        enum_descriptions = value.get("x-enum-descriptions")
        if isinstance(enum_descriptions, dict):
            found.extend(
                (f"{path}.x-enum-descriptions.{enum_value}", text)
                for enum_value, text in enum_descriptions.items()
                if isinstance(text, str)
            )
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                found.extend(_presentation_text(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                found.extend(_presentation_text(child, f"{path}[{index}]"))
    return found


async def test_list_capabilities(client: AsyncClient) -> None:
    assert {value.value for value in PrimaryAssetApplicability} == {
        "required", "optional", "forbidden"
    }
    assert {value.value for value in ProjectContextApplicability} == {
        "required", "derived"
    }
    resp = await client.get(f"{API}/capabilities")
    assert resp.status_code == 200
    caps = resp.json()
    by_id = {c["id"]: c for c in caps}
    ids = set(by_id)
    # A representative sample of the registered capabilities (14 AI + tools).
    assert {"image.caption", "audio.transcribe", "video.upscale"} <= ids
    assert {"video.trim", "image.resize", "audio.normalize"} <= ids
    assert {"image.slideshow", "media.titlecard", "video.compose", "video.lowerthird"} <= ids
    assert "project.export" in ids
    assert "image.kenburns" in ids
    assert len(caps) == 41
    assert {layer.value for layer in CapabilityLayer} == {
        "deterministic_media", "ai_assisted", "composition", "agent_orchestration"
    }
    expected_metadata = {
        "video.trim": ("deterministic_media", "FFmpeg"),
        "image.resize": ("deterministic_media", "ImageMagick"),
        "image.upscale": ("ai_assisted", "Real-ESRGAN (CUDA)"),
        "audio.denoise": ("ai_assisted", "FFmpeg with RNNoise"),
        "image.slideshow": ("composition", "FFmpeg and ImageMagick"),
        "project.export": ("composition", "Internal composition renderer (FFmpeg)"),
    }
    for capability_id, expected in expected_metadata.items():
        assert (by_id[capability_id]["layer"], by_id[capability_id]["tool"]) == expected
    allowed_tools = {
        "FFmpeg",
        "ImageMagick",
        "Real-ESRGAN (CUDA)",
        "FFmpeg with RNNoise",
        "FFmpeg and ImageMagick",
        "OpenCV and FFmpeg",
        "ImageMagick and FFmpeg",
        "Internal composition renderer (FFmpeg)",
        "PySceneDetect",
        "Configured inference backend",
        "Not implemented",
    }
    for capability in caps:
        assert capability["layer"] in {layer.value for layer in CapabilityLayer}
        assert capability["tool"] in allowed_tools
        applicability = capability["applicability"]
        assert applicability["primary_asset"] in {"required", "optional", "forbidden"}
        assert applicability["project_context"] in {"required", "derived"}
        if applicability["primary_asset"] == "forbidden":
            assert capability["supported_media_types"] == []
        else:
            assert capability["supported_media_types"]
        assert "x-ui" not in str(capability)
    for capability_id in ("audio.normalize", "audio.gain", "audio.fade", "audio.denoise"):
        assert by_id[capability_id]["supported_media_types"] == ["video", "audio"]
        assert "video input preserves the video stream" in by_id[capability_id]["description"]
    extract = by_id["video.audio.extract"]
    assert extract["supported_media_types"] == ["video"]
    assert "separate AAC/M4A audio asset" in extract["description"]

    by_id = {capability["id"]: capability for capability in caps}
    assert by_id["video.trim"]["applicability"] == {
        "primary_asset": "required",
        "project_context": "derived",
    }
    assert by_id["image.resize"]["applicability"]["primary_asset"] == "required"
    assert by_id["audio.gain"]["applicability"]["primary_asset"] == "required"
    for capability_id in (
        "video.concat", "audio.concat", "image.slideshow", "video.compose"
    ):
        assert by_id[capability_id]["applicability"]["primary_asset"] == "optional"
    for capability_id in (
        "audio.tts", "audio.music.generate", "media.titlecard", "project.export"
    ):
        assert by_id[capability_id]["applicability"] == {
            "primary_asset": "forbidden",
            "project_context": "required",
        }
        assert by_id[capability_id]["supported_media_types"] == []


async def test_public_capability_presentation_omits_internal_parameter_paths(
    client: AsyncClient,
) -> None:
    response = await client.get(f"{API}/capabilities")
    assert response.status_code == 200
    capabilities = response.json()
    assert len(capabilities) == 41

    for capability in capabilities:
        texts = [("description", capability.get("description", ""))]
        texts.extend(_presentation_text(capability.get("input_schema", {}), "input_schema"))
        texts.extend(_presentation_text(capability.get("output_schema", {}), "output_schema"))
        for field, text in texts:
            assert _INTERNAL_PRESENTATION_REFERENCE.search(text) is None, (
                capability["id"], field, text
            )

        detail = await client.get(f"{API}/capabilities/{capability['id']}")
        assert detail.status_code == 200
        assert detail.json()["description"] == capability["description"]
        assert detail.json()["input_schema"] == capability["input_schema"]

    by_id = {capability["id"]: capability for capability in capabilities}
    assert "asset_ids" in by_id["video.concat"]["input_schema"]["properties"]
    assert "asset_ids" in by_id["audio.concat"]["input_schema"]["properties"]
    assert "asset_ids" in by_id["image.slideshow"]["input_schema"]["properties"]
    assert "subtitle_asset_id" in by_id["video.subtitle.embed"]["input_schema"]["properties"]
    assert "music_asset_id" in by_id["audio.mix"]["input_schema"]["properties"]
    assert "audio_asset_id" in by_id["video.compose"]["input_schema"]["properties"]


async def test_get_capability_detail(client: AsyncClient) -> None:
    resp = await client.get(f"{API}/capabilities/image.caption")
    assert resp.status_code == 200
    cap = resp.json()
    assert cap["id"] == "image.caption"
    assert cap["cost_class"] == "hosted_ai"
    assert cap["layer"] == "ai_assisted"
    assert cap["tool"] == "Configured inference backend"
    assert "image" in cap["supported_media_types"]


def test_capability_library_openapi_is_additive() -> None:
    from app.main import app

    document = app.openapi()
    properties = document["components"]["schemas"]["CapabilityRead"]["properties"]
    layer_schema = properties["layer"]
    if "$ref" in layer_schema:
        layer_name = layer_schema["$ref"].rsplit("/", maxsplit=1)[-1]
        layer_schema = document["components"]["schemas"][layer_name]
    assert layer_schema["enum"] == [
        "deterministic_media", "ai_assisted", "composition", "agent_orchestration"
    ]
    assert properties["tool"]["type"] == "string"
    parameters = document["paths"]["/api/v1/capabilities"]["get"]["parameters"]
    include_disabled = next(item for item in parameters if item["name"] == "include_disabled")
    assert include_disabled["required"] is False
    assert include_disabled["schema"]["default"] is False


async def test_video_transcode_schema_exposes_supported_dependent_choices(
    client: AsyncClient,
) -> None:
    response = await client.get(f"{API}/capabilities/video.transcode")
    assert response.status_code == 200
    schema = response.json()["input_schema"]
    properties = schema["properties"]

    assert properties["format"]["enum"] == ["mp4", "mov", "mkv", "webm"]
    assert properties["video_codec"]["enum"] == ["libx264", "libx265", "libvpx-vp9"]
    assert properties["audio_codec"]["enum"] == [
        "aac", "libmp3lame", "libopus", "libvorbis", "flac"
    ]
    assert properties["video_codec"]["x-enum-descriptions"]["libx264"] == "H.264 / AVC"
    assert properties["audio_codec"]["x-enum-descriptions"]["libopus"] == "Opus"
    assert properties["video_codec"]["x-option-constraints"] == {
        "depends_on": "format",
        "allowed_values": {
            "mp4": ["libx264"],
            "mov": ["libx264"],
            "mkv": ["libx264", "libx265", "libvpx-vp9"],
            "webm": ["libvpx-vp9"],
        },
    }
    assert schema["required"] == ["format"]
    assert schema["allOf"]

    capability = registry.get("video.transcode")
    for container in properties["format"]["enum"]:
        # Both codec controls stay optional; omission retains FFmpeg defaults.
        validate_capability_input(capability, {"format": container})
    for field in ("video_codec", "audio_codec"):
        choices = properties[field]["enum"]
        by_container = properties[field]["x-option-constraints"]["allowed_values"]
        for choice in choices:
            container = next(name for name, values in by_container.items() if choice in values)
            validate_capability_input(capability, {"format": container, field: choice})


async def test_capability_schema_exposes_execution_and_semantic_contract(
    client: AsyncClient,
) -> None:
    mix = (await client.get(f"{API}/capabilities/audio.mix")).json()["input_schema"]
    volume = mix["properties"]["music_volume"]
    assert volume["title"] == "Music volume"
    assert volume["default"] == 0.3
    assert volume["x-unit"] == "ratio"
    music = mix["properties"]["music_asset_id"]
    assert music["x-media-asset"] == {
        "role": "background_audio",
        "media_types": ["audio"],
    }
    assert mix["properties"]["mode"]["x-enum-descriptions"]["duck"]

    concat = (await client.get(f"{API}/capabilities/video.concat")).json()[
        "input_schema"
    ]
    assert concat["properties"]["asset_ids"]["minItems"] == 2

    image_format = (await client.get(f"{API}/capabilities/image.format")).json()[
        "input_schema"
    ]
    assert image_format["properties"]["format"]["enum"] == [
        "jpeg",
        "jpg",
        "png",
        "webp",
        "tiff",
    ]


async def test_get_unknown_capability_404(client: AsyncClient) -> None:
    resp = await client.get(f"{API}/capabilities/does.not.exist")
    assert resp.status_code == 404
