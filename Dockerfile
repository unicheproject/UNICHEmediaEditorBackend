# Base image pinned by tag + digest for reproducible builds (Python 3.12.13).
FROM python:3.12-slim-bookworm@sha256:76d4b7b6305788c6b4c6a19d6a22a3921bf802e9af4d5e1e5bd771208dba74bf

# uv pinned by exact version (not :latest) for reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:0.11.20 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

# System tools for deterministic media capabilities, pinned to bookworm versions.
# ffmpeg -> all video + audio ops; imagemagick -> image ops (`convert` CLI).
# libgomp1 -> OpenMP runtime needed by PyTorch's CPU-side ops (image.upscale).
# libgl1/libglib2.0-0 -> runtime libs required by opencv-python, a transitive
# dependency of both `scenedetect` (video.shot.detect) and `basicsr`/
# `realesrgan` (image.upscale); this slim base image has neither by default
# and opencv-python fails to import without them.
#
# No Vulkan/EGL packages here anymore (libvulkan1, mesa-vulkan-drivers,
# libegl1): image.upscale no longer uses the ncnn-vulkan binary. See
# assets/realesrgan/README.md for why -- that binary produced corrupted
# multi-tile output on real GPU hardware, replaced with the official
# PyTorch/CUDA Real-ESRGAN implementation (no Vulkan involved at all).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg=7:5.1.9-0+deb12u1 \
        imagemagick=8:6.9.11.60+dfsg-1.6+deb12u11 \
        fonts-dejavu-core=2.37-6 \
        libgomp1=12.2.0-14+deb12u1 \
        libgl1=1.6.0-1 \
        libglib2.0-0=2.74.6-2+deb12u9 \
    && rm -rf /var/lib/apt/lists/*

# RNNoise model for the arnndn filter (audio.denoise) — this ffmpeg build has
# no bundled default model, so one must be vendored. See assets/rnnoise/README.md.
RUN mkdir -p /usr/share/rnnoise
COPY assets/rnnoise/sh.rnnn /usr/share/rnnoise/model.rnnn

# Real-ESRGAN weights (image.upscale) — see assets/realesrgan/README.md for
# provenance/license. GPU-only, no ncnn/Vulkan binary involved (see above).
RUN mkdir -p /opt/realesrgan
COPY assets/realesrgan/RealESRGAN_x4plus.pth /opt/realesrgan/RealESRGAN_x4plus.pth

WORKDIR /app

# Install dependencies from the fully-pinned lockfile (with hashes), then the
# project itself so `app` is importable for api/worker/tests regardless of cwd.
# torch/torchvision (and their nvidia-*/triton transitive deps, elsewhere in
# this file) come from PyTorch's own cu121 index; everything else from PyPI.
# --index-strategy unsafe-best-match is required here: uv's default strategy
# locks onto the first index that has ANY version of a package, and PyTorch's
# index happens to also mirror common packages like certifi -- just not our
# pinned version -- which made resolution fail outright instead of falling
# through to PyPI. Safe in this case since both indexes are fully trusted
# (PyPI + the official PyTorch project), unlike the dependency-confusion
# scenario this default guards against.
COPY requirements.lock pyproject.toml ./
RUN uv pip install --system --require-hashes \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --index-strategy unsafe-best-match \
    -r requirements.lock

# basicsr 1.4.2 (unmaintained since 2022) imports
# `torchvision.transforms.functional_tensor`, which torchvision removed in
# 0.17 (the function it needs, rgb_to_grayscale, moved to
# `torchvision.transforms.functional` unchanged) -- patch the one import line
# rather than pin to an old torchvision, which would also force an old torch
# without current CUDA/driver support.
RUN sed -i \
    's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /usr/local/lib/python3.12/site-packages/basicsr/data/degradations.py

COPY . .
RUN uv pip install --system --no-deps .

# Storage dir is mounted as a volume in compose; create a default for standalone runs
RUN mkdir -p /data/storage

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
