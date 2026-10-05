from httpx import AsyncClient

from app.capabilities import registry
from app.capabilities.validation import validate_capability_input
from app.schemas.capability import (
    PrimaryAssetApplicability,
    ProjectContextApplicability,
)

API = "/api/v1"


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
    for capability in caps:
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


async def test_get_capability_detail(client: AsyncClient) -> None:
    resp = await client.get(f"{API}/capabilities/image.caption")
    assert resp.status_code == 200
    cap = resp.json()
    assert cap["id"] == "image.caption"
    assert cap["cost_class"] == "hosted_ai"
    assert "image" in cap["supported_media_types"]


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
