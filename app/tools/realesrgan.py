"""Real-ESRGAN (official PyTorch/CUDA implementation) for GPU image upscaling.

Replaces the previously vendored `realesrgan-ncnn-vulkan` binary, which had a
genuine multi-tile correctness bug on real GPU hardware (confirmed via
extensive isolation testing on 2026-07-16 -- see workspace CLAUDE.md project
history and assets/realesrgan/README.md). Requires a CUDA GPU; per workspace
CLAUDE.md policy this deliberately does not fall back to CPU.

The model (weights + architecture) is unchanged -- same author, same
`realesrgan-x4plus` family -- only the runtime (PyTorch/cuDNN instead of
ncnn/Vulkan) changed.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from app.core.errors import ToolError

MODEL_PATH = "/opt/realesrgan/RealESRGAN_x4plus.pth"
NET_SCALE = 4
TILE_SIZE = 400  # verified artifact-free on an 8GB GTX 1080; see README

_VALID_SCALES = {2, 3, 4}

# RealESRGANer.enhance() stores intermediate state (img/output) as instance
# attributes, not thread-locals -- concurrent calls on the same instance from
# different threads (arq can run several jobs at once) would corrupt each
# other's state. One GPU has no real parallelism to give up anyway.
_lock = threading.Lock()
_upsampler: Any = None


def _get_upsampler() -> Any:
    global _upsampler
    if _upsampler is not None:
        return _upsampler

    import torch

    if not torch.cuda.is_available():
        raise ToolError(
            "image.upscale requires a CUDA GPU and none is visible to this "
            "worker (no CPU fallback by design -- see workspace CLAUDE.md)"
        )

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=NET_SCALE
    )
    _upsampler = RealESRGANer(
        scale=NET_SCALE,
        model_path=MODEL_PATH,
        model=model,
        tile=TILE_SIZE,
        tile_pad=10,
        pre_pad=0,
        half=False,  # this GPU generation has weak native fp16 arithmetic
        gpu_id=0,
    )
    return _upsampler


def _upscale_sync(src: str, dst: str, scale: int) -> None:
    import cv2

    with _lock:
        upsampler = _get_upsampler()
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ToolError(f"image.upscale could not read input image: {src}")
        try:
            output, _ = upsampler.enhance(img, outscale=scale)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"image.upscale failed: {exc}") from exc
        if not cv2.imwrite(dst, output):
            raise ToolError(f"image.upscale could not write output image: {dst}")


async def upscale(src: str, dst: str, scale: int | None) -> None:
    scale = scale if scale is not None else 4
    if scale not in _VALID_SCALES:
        raise ToolError(
            f"image.upscale 'scale' must be one of {sorted(_VALID_SCALES)}, got {scale!r}"
        )
    await asyncio.to_thread(_upscale_sync, src, dst, scale)
