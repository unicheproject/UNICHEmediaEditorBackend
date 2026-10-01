from httpx import AsyncClient

API = "/api/v1"


async def test_list_capabilities(client: AsyncClient) -> None:
    resp = await client.get(f"{API}/capabilities")
    assert resp.status_code == 200
    caps = resp.json()
    by_id = {c["id"]: c for c in caps}
    ids = set(by_id)
    # A representative sample of the registered capabilities (14 AI + 20 tools).
    assert {"image.caption", "audio.transcribe", "video.upscale"} <= ids
    assert {"video.trim", "image.resize", "audio.normalize"} <= ids
    assert {"image.slideshow", "media.titlecard", "video.compose", "video.lowerthird"} <= ids
    assert "image.kenburns" in ids
    assert len(caps) == 40
    for capability_id in ("audio.normalize", "audio.gain", "audio.fade", "audio.denoise"):
        assert by_id[capability_id]["supported_media_types"] == ["video", "audio"]
        assert "video input preserves the video stream" in by_id[capability_id]["description"]
    extract = by_id["video.audio.extract"]
    assert extract["supported_media_types"] == ["video"]
    assert "separate AAC/M4A audio asset" in extract["description"]


async def test_get_capability_detail(client: AsyncClient) -> None:
    resp = await client.get(f"{API}/capabilities/image.caption")
    assert resp.status_code == 200
    cap = resp.json()
    assert cap["id"] == "image.caption"
    assert cap["cost_class"] == "hosted_ai"
    assert "image" in cap["supported_media_types"]


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
