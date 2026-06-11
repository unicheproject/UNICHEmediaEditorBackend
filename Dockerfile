FROM python:3.12-slim

# Faster, reproducible installs via uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy project metadata + source, then install (project + dev deps) into the system env.
# Installing the project itself (not just deps) keeps `app` importable for api/worker/tests.
COPY . .
RUN uv pip install --system ".[dev]"

# Storage dir is mounted as a volume in compose; create a default for standalone runs
RUN mkdir -p /data/storage

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
