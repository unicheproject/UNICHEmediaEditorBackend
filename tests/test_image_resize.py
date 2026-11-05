"""image.resize: optional aspect-preserving side, first frame and orientation."""

from __future__ import annotations

import shutil
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
from tests.test_image_crop import _image, _size, _with_exif_orientation

API = "/api/v1"
needs_convert = pytest.mark.skipif(
    shutil.which("convert") is None, reason="ImageMagick not installed"
)


def test_schema_needs_one_side_and_reuse_is_bumped() -> None:
    cap = registry.get("image.resize")
    for params in ({"width": 100}, {"height": 50}, {"width": 100, "height": 50}):
        validate_capability_input(cap, params)
    for params in ({}, {"width": 0}, {"height": -1}):
        with pytest.raises(CapabilityInputValidationError):
            validate_capability_input(cap, params)
    assert REUSE_VERSIONS["image.resize"] == "2"


async def test_tool_requires_a_side() -> None:
    with pytest.raises(ValidationError, match="width, a height"):
        await imagemagick.resize("in.png", "out.png")


@needs_convert
@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [(150, None, (150, 100)), (None, 100, (150, 100)), (151, None, (151, 101)),
     (120, 120, (120, 120))],  # both sides: exact size, aspect not kept
)
async def test_resize_sizes(
    tmp_path: Path, width: int | None, height: int | None, expected: tuple[int, int]
) -> None:
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    _image(src, "300x200")
    await imagemagick.resize(str(src), str(dst), width, height)
    assert _size(dst) == expected


@needs_convert
async def test_multi_page_input_resizes_the_first_page(tmp_path: Path) -> None:
    src, dst = tmp_path / "pages.tiff", tmp_path / "out.png"
    subprocess.run(["convert", "-size", "100x100", "xc:red", "xc:blue", str(src)],
                   check=True, capture_output=True)
    await imagemagick.resize(str(src), str(dst), 50, None)
    assert dst.is_file() and _size(dst) == (50, 50)
    assert not list(tmp_path.glob("out-*.png"))


@needs_convert
async def test_orientation_is_applied_before_resizing(tmp_path: Path) -> None:
    plain = tmp_path / "plain.jpg"
    _image(plain, "300x200", "blue")
    rotated = tmp_path / "rotated.jpg"
    rotated.write_bytes(_with_exif_orientation(plain.read_bytes(), 6))  # displayed 200×300
    dst = tmp_path / "out.png"
    await imagemagick.resize(str(rotated), str(dst), 100, None)
    assert _size(dst) == (100, 150)


@needs_convert
async def test_width_only_resize_job_succeeds(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "photo.png"
    _image(src, "300x200")
    pid = await make_project("image-resize")
    resp = await client.post(
        f"{API}/projects/{pid}/assets", files={"file": ("photo.png", src.read_bytes(), "image/png")}
    )
    aid = resp.json()["id"]
    async with async_session_factory() as session:
        await probe_asset_metadata(session, uuid.UUID(aid))
    job_resp = await client.post(f"{API}/jobs", json={
        "capability_id": "image.resize", "asset_id": aid, "input": {"width": 75},
    })
    assert job_resp.status_code == 201, job_resp.text
    job = (await client.get(f"{API}/jobs/{job_resp.json()['id']}")).json()
    assert job["status"] == "succeeded", job
    out = tmp_path / "out.png"
    out.write_bytes((await client.get(job["output"]["outputs"][0]["download_path"])).content)
    assert _size(out) == (75, 50)
