# Real-ESRGAN weights (PyTorch/CUDA) for `image.upscale`

`RealESRGAN_x4plus.pth` is the official general-purpose photo model from
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (Wang et al., 2021),
distributed under the **BSD 3-Clause license** by the author (xinntao/Xintao
Wang). Downloaded from the
[`xinntao/Real-ESRGAN` v0.1.0 release](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0)
(`RealESRGAN_x4plus.pth`, sha256
`4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`).

## Why PyTorch/CUDA, not the ncnn/Vulkan binary

This capability originally used the vendored `realesrgan-ncnn-vulkan` binary
(ncnn/Vulkan implementation, same model). On 2026-07-16, once real NVIDIA GPU
passthrough was working in production, that binary was found to have a
genuine correctness bug: multi-tile output was corrupted ("puzzle"-tiled) when
run against a real Vulkan GPU device, reproducible regardless of tile size,
thread concurrency (`-j`), or TTA mode — confirmed via extensive isolation
testing (see workspace `CLAUDE.md` project history). Only a single tile
(no splitting) was clean, and the GPU's 8GB VRAM isn't enough to avoid tiling
for realistic photo sizes. The ncnn-vulkan project has had no release since
April 2022, so there was no upstream fix to pick up.

Falling back to the CPU/Mesa software path to dodge this is explicitly
forbidden (see workspace `CLAUDE.md`, "GPU-accelerated capabilities must
actually run on the GPU"). Instead, this capability now uses the official
PyTorch implementation of the *same* model (same author, same architecture
and weights family) via the `realesrgan` + `basicsr` PyPI packages, running
on CUDA. `RealESRGANer.tile_process()` does the same tile-and-stitch
approach, but as ordinary tensor padding/blending on top of cuDNN rather than
hand-written Vulkan compute shaders — a far more widely used and tested code
path across NVIDIA GPU generations. Verified directly on the production GPU
(GTX 1080) against the exact input that corrupted the ncnn binary: clean
output, no tiling artifacts.

## Requires a CUDA GPU — no CPU fallback

Per workspace `CLAUDE.md` policy, `app/tools/realesrgan.py` raises `ToolError`
if no CUDA device is visible rather than silently running on CPU. This means
`image.upscale` only works on a host with a real NVIDIA GPU passed through
(CDI device passthrough — see `DEPLOYMENT.md`); it cannot be exercised
locally on a machine without one. `half=False` (full FP32) is used
deliberately: this GPU generation (Pascal) has weak native FP16 arithmetic
support, which is an unrelated but real risk of numeric corruption in FP16
inference paths.
