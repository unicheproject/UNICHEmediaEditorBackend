"""image.crop: bounds, EXIF orientation and first-frame semantics."""

from __future__ import annotations

import shutil
import struct
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.agent.reuse import REUSE_VERSIONS
from app.capabilities import registry
from app.capabilities.validation import (
    CapabilityInputValidationError,
    validate_capability_input,
)
from app.core.database import async_session_factory
from app.core.errors import ValidationError
from app.services.media_metadata import probe_asset_metadata
from app.tools import imagemagick

API = "/api/v1"
HAVE_CONVERT = shutil.which("convert") is not None
needs_convert = pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")


def test_schema_rejects_negative_offsets_and_reuse_is_bumped() -> None:
    cap = registry.get("image.crop")
    for params in ({"x": -1, "y": 0, "width": 5, "height": 5},
                   {"x": 0, "y": -3, "width": 5, "height": 5}):
        with pytest.raises(CapabilityInputValidationError):
            validate_capability_input(cap, params)
    assert REUSE_VERSIONS["image.crop"] == "2"


def _image(path: Path, size: str = "300x200", colour: str = "red") -> None:
    subprocess.run(["convert", "-size", size, f"xc:{colour}", str(path)],
                   check=True, capture_output=True)


def _with_exif_orientation(jpeg: bytes, orientation: int) -> bytes:
    """Insert a minimal big-endian EXIF APP1 segment carrying one Orientation tag."""
    entry = struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0)
    tiff = b"MM\x00*" + struct.pack(">I", 8) + struct.pack(">H", 1) + entry + b"\x00" * 4
    payload = b"Exif\x00\x00" + tiff
    return jpeg[:2] + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + jpeg[2:]


def _size(path: Path) -> tuple[int, int]:
    out = subprocess.run(["identify", "-format", "%w %h", str(path)],
                         check=True, capture_output=True, text=True).stdout
    width, height = out.split()
    return int(width), int(height)


@needs_convert
@pytest.mark.parametrize(
    ("rect", "message"),
    [((100, 100, 250, 150), r"fit inside the image \(300×200\)"),  # IM clipped to 200×100
     ((400, 400, 50, 50), "fit inside"),  # IM returned a 1×1 image
     ((-10, 0, 50, 50), "negative")],
)
async def test_out_of_frame_rectangles_are_rejected(
    tmp_path: Path, rect: tuple[int, int, int, int], message: str
) -> None:
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _image(src)
    with pytest.raises(ValidationError, match=message):
        await imagemagick.crop(str(src), str(dst), *rect)
    assert not dst.exists()


@needs_convert
async def test_valid_crop_has_exact_size(tmp_path: Path) -> None:
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _image(src)
    await imagemagick.crop(str(src), str(dst), 50, 20, 250, 180)
    assert _size(dst) == (250, 180)


@needs_convert
async def test_multi_page_input_crops_the_first_page(tmp_path: Path) -> None:
    src, dst = tmp_path / "pages.tiff", tmp_path / "out.png"
    subprocess.run(["convert", "-size", "100x100", "xc:red", "xc:blue", str(src)],
                   check=True, capture_output=True)
    await imagemagick.crop(str(src), str(dst), 0, 0, 50, 50)
    assert dst.is_file() and _size(dst) == (50, 50)
    assert not list(tmp_path.glob("out-*.png"))
    colour = subprocess.run(["convert", str(dst), "-format", "%[pixel:p{1,1}]", "info:"],
                            check=True, capture_output=True, text=True).stdout
    assert "red" in colour or "255,0,0" in colour


@needs_convert
async def test_exif_orientation_defines_the_coordinate_frame(tmp_path: Path) -> None:
    plain = tmp_path / "plain.jpg"
    _image(plain, "300x200", "blue")
    rotated = tmp_path / "rotated.jpg"
    rotated.write_bytes(_with_exif_orientation(plain.read_bytes(), 6))  # displayed 200×300
    assert await imagemagick.display_size(str(rotated)) == (200, 300)
    dst = tmp_path / "out.png"
    with pytest.raises(ValidationError, match=r"\(200×300\)"):
        await imagemagick.crop(str(rotated), str(dst), 0, 0, 300, 200)
    await imagemagick.crop(str(rotated), str(dst), 0, 50, 200, 250)
    assert _size(dst) == (200, 250)


async def _upload(client: AsyncClient, pid: str, path: Path, mime: str) -> str:
    resp = await client.post(
        f"{API}/projects/{pid}/assets", files={"file": (path.name, path.read_bytes(), mime)}
    )
    assert resp.status_code == 201, resp.text
    asset_id: str = resp.json()["id"]
    async with async_session_factory() as session:
        await probe_asset_metadata(session, uuid.UUID(asset_id))
    return asset_id


@needs_convert
async def test_admission_rejects_and_accepts_by_orientation(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    plain = tmp_path / "plain.jpg"
    _image(plain, "300x200", "blue")
    rotated = tmp_path / "rotated.jpg"
    rotated.write_bytes(_with_exif_orientation(plain.read_bytes(), 6))
    pid = await make_project("image-crop")
    plain_id = await _upload(client, pid, plain, "image/jpeg")
    rotated_id = await _upload(client, pid, rotated, "image/jpeg")

    rejected = await client.post(f"{API}/jobs", json={
        "capability_id": "image.crop", "asset_id": plain_id,
        "input": {"x": 0, "y": 0, "width": 400, "height": 400},
    })
    assert rejected.status_code == 422, rejected.text
    assert "fit inside the image (300×200)" in rejected.json()["error"]["message"]

    # Fits only the displayed (rotated) frame: admitted, exact check at execution.
    accepted = await client.post(f"{API}/jobs", json={
        "capability_id": "image.crop", "asset_id": rotated_id,
        "input": {"x": 0, "y": 50, "width": 200, "height": 250},
    })
    assert accepted.status_code == 201, accepted.text
    job = (await client.get(f"{API}/jobs/{accepted.json()['id']}")).json()
    assert job["status"] == "succeeded", job
