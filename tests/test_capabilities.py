from httpx import AsyncClient

API = "/api/v1"


async def test_list_capabilities(client: AsyncClient) -> None:
    resp = await client.get(f"{API}/capabilities")
    assert resp.status_code == 200
    caps = resp.json()
    ids = {c["id"] for c in caps}
    # A representative sample of the 14 registered capabilities.
    assert {"image.caption", "audio.transcribe", "video.upscale"} <= ids
    assert len(caps) == 14


async def test_get_capability_detail(client: AsyncClient) -> None:
    resp = await client.get(f"{API}/capabilities/image.caption")
    assert resp.status_code == 200
    cap = resp.json()
    assert cap["id"] == "image.caption"
    assert cap["cost_class"] == "hosted_ai"
    assert "image" in cap["supported_media_types"]


async def test_get_unknown_capability_404(client: AsyncClient) -> None:
    resp = await client.get(f"{API}/capabilities/does.not.exist")
    assert resp.status_code == 404
