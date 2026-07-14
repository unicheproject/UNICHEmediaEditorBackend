"""Real-ESRGAN wrapper (ncnn-vulkan binary) for deterministic image upscaling.

The binary + model weights are vendored under assets/realesrgan/ (see its
README for provenance/license) and copied into the Docker image at
/opt/realesrgan/. `-g 0` always selects Vulkan device 0: with the host GPU
passed through (docker-compose `worker.devices: - /dev/dri:/dev/dri`, Linux
only) that's real hardware and upscaling takes seconds; without it, Mesa's
llvmpipe software rasterizer is the only device and still works, just tens of
minutes slower per image.
"""

from __future__ import annotations

from app.core.errors import ToolError
from app.tools import runner

REALESRGAN_BIN = "/opt/realesrgan/realesrgan-ncnn-vulkan"
REALESRGAN_MODELS_DIR = "/opt/realesrgan/models"
DEFAULT_MODEL = "realesrgan-x4plus"
GPU_DEVICE_ID = "0"

# Generous timeout: falls back to llvmpipe (much slower) with no GPU passthrough.
UPSCALE_TIMEOUT = 900.0

_VALID_SCALES = {2, 3, 4}


async def upscale(src: str, dst: str, scale: int | None) -> None:
    scale = scale if scale is not None else 4
    if scale not in _VALID_SCALES:
        raise ToolError(
            f"image.upscale 'scale' must be one of {sorted(_VALID_SCALES)}, got {scale!r}"
        )
    await runner.run(
        [
            REALESRGAN_BIN,
            "-i", src,
            "-o", dst,
            "-m", REALESRGAN_MODELS_DIR,
            "-n", DEFAULT_MODEL,
            "-s", str(scale),
            "-g", GPU_DEVICE_ID,
        ],
        timeout=UPSCALE_TIMEOUT,
    )
