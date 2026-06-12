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
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg=7:5.1.9-0+deb12u1 \
        imagemagick=8:6.9.11.60+dfsg-1.6+deb12u10 \
        fonts-dejavu-core=2.37-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies from the fully-pinned lockfile (with hashes), then the
# project itself so `app` is importable for api/worker/tests regardless of cwd.
COPY requirements.lock pyproject.toml ./
RUN uv pip install --system --require-hashes -r requirements.lock
COPY . .
RUN uv pip install --system --no-deps .

# Storage dir is mounted as a volume in compose; create a default for standalone runs
RUN mkdir -p /data/storage

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
