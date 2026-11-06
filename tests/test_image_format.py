"""image.format: stated quality, lossless choices and transparency (devfixes C-11)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.agent.reuse import REUSE_VERSIONS
from app.capabilities import registry
from app.capabilities.validation import (
    CapabilityInputValidationError,
    validate_capability_input,
)
from tests.test_image_crop import _image, _with_exif_orientation

API = "/api/v1"
HAVE_CONVERT = shutil.which("convert") is not None


def _identify(path: Path, fmt: str) -> str:
    return subprocess.run(
        ["identify", "-format", fmt, f"{path}[0]"], check=True, capture_output=True, text=True
    ).stdout.strip()


# --- contract ------------------------------------------------------------------


def test_format_choices_are_unique_and_described() -> None:
    schema = registry.get("image.format").input_schema
    fmt = schema["properties"]["format"]
    assert fmt["enum"] == ["jpeg", "png", "webp", "tiff"]  # no jpeg/jpg duplicate
    assert set(fmt["x-enum-descriptions"]) == set(fmt["enum"])
    assert schema["properties"]["quality"]["minimum"] == 1
    assert schema["properties"]["quality"]["maximum"] == 100
    assert REUSE_VERSIONS["image.format"] == "2"


@pytest.mark.parametrize(
    "params",
    [
        {"format": "jpeg", "quality": 80},
        {"format": "webp", "quality": 60},
        {"format": "webp", "lossless": True},
        {"format": "png", "flatten_transparency": True},
        {"format": "tiff"},
    ],
)
def test_valid_combinations(params: dict) -> None:
    validate_capability_input(registry.get("image.format"), params)


@pytest.mark.parametrize(
    "params",
    [
        {"format": "jpg"},
        {"format": "png", "quality": 80},
        {"format": "tiff", "quality": 80},
        {"format": "jpeg", "lossless": True},
        {"format": "webp", "lossless": True, "quality": 80},
        {"format": "jpeg", "quality": 0},
        {"format": "jpeg", "quality": 101},
        {"format": "svg"},
    ],
)
def test_invalid_combinations(params: dict) -> None:
    with pytest.raises(CapabilityInputValidationError):
        validate_capability_input(registry.get("image.format"), params)


# --- real ImageMagick through Jobs ------------------------------------------------


async def _convert(
    client: AsyncClient, make_project, src: Path, mime: str, params: dict, tmp: Path
) -> tuple[dict, Path]:
    pid = await make_project(f"format-{params['format']}")
    resp = await client.post(
        f"{API}/projects/{pid}/assets", files={"file": (src.name, src.read_bytes(), mime)}
    )
    assert resp.status_code == 201, resp.text
    job = await client.post(f"{API}/jobs", json={
        "capability_id": "image.format", "asset_id": resp.json()["id"], "input": params,
    })
    assert job.status_code == 201, job.text
    body = (await client.get(f"{API}/jobs/{job.json()['id']}")).json()
    assert body["status"] == "succeeded", body
    output = body["output"]["outputs"][0]
    path = tmp / f"{output['asset_id']}_{output['filename']}"
    path.write_bytes((await client.get(output["download_path"])).content)
    return body["output"], path


def _noisy_tiff(path: Path) -> None:
    subprocess.run(
        ["convert", "-size", "400x300", "-seed", "7", "plasma:fractal", "-depth", "8",
         str(path)],
        check=True, capture_output=True,
    )


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
async def test_webp_quality_is_stated_and_controls_size(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "master.tiff"
    _noisy_tiff(src)
    default, default_path = await _convert(
        client, make_project, src, "image/tiff", {"format": "webp"}, tmp_path
    )
    assert default["quality"] == 90 and default["lossless"] is False
    low, low_path = await _convert(
        client, make_project, src, "image/tiff", {"format": "webp", "quality": 30}, tmp_path
    )
    assert low["quality"] == 30
    assert low_path.stat().st_size < default_path.stat().st_size
    lossless, lossless_path = await _convert(
        client, make_project, src, "image/tiff", {"format": "webp", "lossless": True},
        tmp_path,
    )
    assert lossless["lossless"] is True and "quality" not in lossless
    assert lossless_path.stat().st_size > default_path.stat().st_size
    # Lossless WebP keeps every pixel of the master.
    diff = subprocess.run(
        ["compare", "-metric", "AE", str(src), str(lossless_path), "null:"],
        capture_output=True, text=True,
    )
    assert diff.stderr.strip() == "0"


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
async def test_jpeg_quality_and_tiff_is_lossless(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "photo.jpg"
    _image(src, "300x200", "blue")
    output, path = await _convert(
        client, make_project, src, "image/jpeg", {"format": "jpeg", "quality": 55}, tmp_path
    )
    assert output["quality"] == 55
    assert _identify(path, "%Q") == "55"
    output, path = await _convert(
        client, make_project, src, "image/jpeg", {"format": "tiff"}, tmp_path
    )
    assert output["lossless"] is True
    # A JPEG source must not carry JPEG compression into the "archival" TIFF.
    assert _identify(path, "%C") == "LZW"


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
async def test_transparency_is_kept_or_flattened(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    src = tmp_path / "logo.png"
    subprocess.run(
        ["convert", "-size", "100x100", "xc:none", "-fill", "red", "-draw",
         "rectangle 25,25 74,74", str(src)],
        check=True, capture_output=True,
    )
    _, kept = await _convert(
        client, make_project, src, "image/png", {"format": "webp", "lossless": True},
        tmp_path,
    )
    assert _identify(kept, "%[pixel:p{0,0}]") in {"srgba(0,0,0,0)", "none"}
    _, flat = await _convert(
        client, make_project, src, "image/png",
        {"format": "png", "flatten_transparency": True}, tmp_path,
    )
    assert _identify(flat, "%[pixel:p{0,0}]") in {"srgb(255,255,255)", "white"}
    _, jpeg = await _convert(client, make_project, src, "image/png", {"format": "jpeg"}, tmp_path)
    # Previously transparent areas came out black in JPEG.
    assert _identify(jpeg, "%[fx:p{0,0}.r>0.95&&p{0,0}.g>0.95&&p{0,0}.b>0.95]") == "1"


@pytest.mark.skipif(not HAVE_CONVERT, reason="ImageMagick not installed")
async def test_multi_page_and_oriented_sources_give_one_upright_image(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pages = tmp_path / "scan.tiff"
    subprocess.run(
        ["convert", "-size", "120x80", "xc:red", "-size", "120x80", "xc:blue", str(pages)],
        check=True, capture_output=True,
    )
    output, path = await _convert(
        client, make_project, pages, "image/tiff", {"format": "png"}, tmp_path
    )
    assert len(output["outputs"]) == 1
    assert _identify(path, "%w %h") == "120 80"

    plain = tmp_path / "plain.jpg"
    _image(plain, "300x200", "green")
    rotated = tmp_path / "phone.jpg"
    rotated.write_bytes(_with_exif_orientation(plain.read_bytes(), 6))
    _, path = await _convert(client, make_project, rotated, "image/jpeg", {"format": "png"},
                             tmp_path)
    assert _identify(path, "%w %h") == "200 300"
