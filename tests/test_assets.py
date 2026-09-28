import io
import shutil
import subprocess

import pytest
from httpx import AsyncClient

API = "/api/v1"

HAVE_IMAGEMAGICK = (
    shutil.which("convert") is not None and shutil.which("identify") is not None
)


def _make_image(path: str) -> None:
    subprocess.run(
        ["convert", "-size", "24x16", "xc:#336699", path],
        check=True,
        capture_output=True,
    )


async def test_upload_list_get_download_delete(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project()

    files = {"file": ("photo.png", io.BytesIO(b"fake-image-bytes"), "image/png")}
    resp = await client.post(f"{API}/projects/{pid}/assets", files=files)
    assert resp.status_code == 201
    asset = resp.json()
    aid = asset["id"]
    assert asset["media_type"] == "image"
    assert asset["extension"] == "png"
    assert asset["size_bytes"] == len(b"fake-image-bytes")
    assert len(asset["checksum_sha256"]) == 64

    # List
    resp = await client.get(f"{API}/projects/{pid}/assets")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    # Metadata
    resp = await client.get(f"{API}/assets/{aid}")
    assert resp.status_code == 200

    # Download
    resp = await client.get(f"{API}/assets/{aid}/download")
    assert resp.status_code == 200
    assert resp.content == b"fake-image-bytes"

    # Rename
    resp = await client.patch(f"{API}/assets/{aid}", json={"original_filename": "vacation.png"})
    assert resp.status_code == 200
    assert resp.json()["original_filename"] == "vacation.png"
    resp = await client.get(f"{API}/assets/{aid}/download")
    assert resp.status_code == 200
    assert resp.headers["content-disposition"].endswith('filename="vacation.png"')

    # Rename rejects an empty name
    resp = await client.patch(f"{API}/assets/{aid}", json={"original_filename": ""})
    assert resp.status_code == 422

    # Soft delete
    resp = await client.delete(f"{API}/assets/{aid}")
    assert resp.status_code == 204
    resp = await client.get(f"{API}/assets/{aid}")
    assert resp.status_code == 404

    # Renaming a deleted (gone) asset 404s
    resp = await client.patch(f"{API}/assets/{aid}", json={"original_filename": "x.png"})
    assert resp.status_code == 404


async def test_upload_rejects_bad_extension(client: AsyncClient, make_project) -> None:
    pid = await make_project()
    files = {"file": ("malware.exe", io.BytesIO(b"x"), "application/octet-stream")}
    resp = await client.post(f"{API}/projects/{pid}/assets", files=files)
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_media_type"


async def test_upload_rejects_oversize(client: AsyncClient, make_project) -> None:
    pid = await make_project()
    # MAX_UPLOAD_SIZE_MB=5 in tests -> exceed it.
    big = b"0" * (6 * 1024 * 1024)
    files = {"file": ("big.wav", io.BytesIO(big), "audio/wav")}
    resp = await client.post(f"{API}/projects/{pid}/assets", files=files)
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


async def test_audio_and_video_media_types(client: AsyncClient, make_project) -> None:
    pid = await make_project()
    for name, mime, expected in [
        ("clip.mp3", "audio/mpeg", "audio"),
        ("clip.mp4", "video/mp4", "video"),
    ]:
        files = {"file": (name, io.BytesIO(b"data"), mime)}
        resp = await client.post(f"{API}/projects/{pid}/assets", files=files)
        assert resp.status_code == 201
        assert resp.json()["media_type"] == expected


@pytest.mark.skipif(not HAVE_IMAGEMAGICK, reason="ImageMagick is not installed")
@pytest.mark.parametrize(
    ("filename", "mime_type"),
    [
        ("photo.png", "image/png"),
        ("photo.tif", "image/tiff"),
        ("photo.tiff", "image/tiff"),
    ],
)
async def test_image_thumbnail_is_a_decodable_jpeg(
    client: AsyncClient, make_project, tmp_path, filename: str, mime_type: str
) -> None:
    source = tmp_path / filename
    _make_image(str(source))
    pid = await make_project()

    with source.open("rb") as stream:
        uploaded = await client.post(
            f"{API}/projects/{pid}/assets",
            files={"file": (filename, stream, mime_type)},
        )
    assert uploaded.status_code == 201, uploaded.text

    thumbnail = await client.get(
        f"{API}/assets/{uploaded.json()['id']}/thumbnail"
    )
    assert thumbnail.status_code == 200, thumbnail.text
    assert thumbnail.headers["content-type"] == "image/jpeg"

    output = tmp_path / f"{filename}.thumbnail.jpg"
    output.write_bytes(thumbnail.content)
    identified = subprocess.run(
        ["identify", "-format", "%m %wx%h", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert identified == "JPEG 24x16"


@pytest.mark.skipif(not HAVE_IMAGEMAGICK, reason="ImageMagick is not installed")
async def test_corrupt_tiff_thumbnail_returns_controlled_tool_error(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project()
    uploaded = await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": ("broken.tiff", io.BytesIO(b"not-a-tiff"), "image/tiff")},
    )
    assert uploaded.status_code == 201

    thumbnail = await client.get(
        f"{API}/assets/{uploaded.json()['id']}/thumbnail"
    )
    assert thumbnail.status_code == 500
    assert thumbnail.json()["error"]["code"] == "tool_error"
