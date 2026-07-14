# Real-ESRGAN (ncnn-vulkan) for `image.upscale`

`realesrgan-ncnn-vulkan` is the ncnn/Vulkan implementation of
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (Wang et al., 2021),
distributed under the **MIT license** by the same author (xinntao). Binary +
model weights vendored here came from the
[`xinntao/Real-ESRGAN` v0.2.5.0 release](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0)
(`realesrgan-ncnn-vulkan-20220424-ubuntu.zip`).

Only `realesrgan-x4plus.{param,bin}` is vendored (the general-purpose photo
model used by `DEFAULT_MODEL` in `app/tools/realesrgan.py`) — the release
also ships anime-specific variants (`realesrgan-x4plus-anime`,
`realesr-animevideov3-*`) that this capability doesn't use, so they're
omitted to keep the image lean.

## Requires a Vulkan device

The binary needs a Vulkan device to run against. Two paths:

- **Real GPU (fast)**: pass the host's `/dev/dri` through to the `worker`
  container (see `docker-compose.yml`, Linux only). A small image upscales in
  well under a minute.
- **No GPU (very slow)**: falls back to Mesa's `llvmpipe` software rasterizer
  (installed via `mesa-vulkan-drivers` in the Dockerfile) — this *works* but
  can take tens of minutes for a single small image. Confirmed by a manual
  spike test: ~31 minutes with software rendering vs. ~34 seconds with the
  host's Intel iGPU passed through, for the same image.

`-g 0` (hardcoded in `app/tools/realesrgan.py`) always selects Vulkan device
0, which is the real GPU when passed through, or the sole `llvmpipe` device
otherwise — no separate CPU-only code path needed.
