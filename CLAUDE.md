# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MVP backend for the UNICHE Media Editor — a browser-based media-editing tool for
cultural-heritage professionals. FastAPI + async SQLAlchemy + Postgres, with an
async job runner (arq/Redis) that executes media/AI operations defined in a
**capability registry**. AI ops route to a hosted-inference **provider
abstraction** (mock by default); deterministic ops shell out to **FFmpeg /
ImageMagick**. Local filesystem storage behind a swappable abstraction.

## Commands

The stack runs in Docker; local tooling runs in a venv.

```bash
# Run the full stack (api + worker + postgres + redis), with hot reload
docker compose up --build
docker compose exec api alembic upgrade head     # apply migrations (first run / after model changes)

# Tests / lint / types — run inside the api container (has ffmpeg+imagemagick),
# so the skip-guarded tool-execution tests actually run:
docker compose exec api pytest
docker compose exec api ruff check .
docker compose exec api mypy app

# ...or locally in the venv:
python3.12 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
pytest                                  # single file: pytest tests/test_jobs.py
                                        # single test: pytest tests/test_jobs.py::test_image_caption_job_succeeds
ruff check . ; mypy app

# Migrations
docker compose exec api alembic revision --autogenerate -m "msg"
docker compose exec api alembic downgrade -1
```

Worker entrypoint: `arq app.workers.worker.WorkerSettings`.

## Critical gotchas

- **api and worker are separate Docker images** built from the same Dockerfile.
  After changing system deps (ffmpeg/imagemagick) or anything affecting the
  image, rebuild **both** (`docker compose build` with no service arg). A common
  bug: jobs fail with "Tool not found: ffmpeg" because only `api` was rebuilt —
  **tool jobs run in the worker**.
- **Dependencies are pinned via `requirements.lock`** (compiled with
  `uv pip compile --extra dev --generate-hashes pyproject.toml -o requirements.lock`)
  and installed with `--require-hashes` in Docker. `pyproject.toml` holds the
  human-readable ranges. After changing deps in `pyproject.toml`, regenerate the
  lock. The Dockerfile also pins the base image (tag+digest), uv, and apt
  package versions — keep these reproducible.
- **Tests use SQLite + run jobs eagerly** (no Redis/arq). `tests/conftest.py`
  sets env vars (DATABASE_URL, STORAGE_DIR, INFERENCE_PROVIDER=mock) **before**
  importing the app, and overrides the `get_enqueuer` dependency so a job runs
  synchronously in-process on creation. Tool-execution tests are
  `skipif`-guarded on `shutil.which("ffmpeg"/"convert")`.
- The cross-dialect `JSONType` (JSONB on Postgres, JSON on SQLite) and `sa.Uuid`
  let the same models run under both Postgres (prod) and SQLite (tests).

## Architecture

Request path never does heavy work. `POST /jobs` validates + persists a `queued`
Job and enqueues `run_job(job_id)` to arq; the **worker** executes it. State
transitions (`queued→running→succeeded/failed`) are committed individually so
polling / the SSE endpoint see live progress.

```
API route → service → enqueue ─▶ Redis ─▶ arq worker → services.jobs.execute_job
                                                              │
                                              capability registry → handler
                                                              │
                                  provider factory (AI)  OR  app/tools/* (FFmpeg/IM)
```

### Capabilities and handlers — the core extension point

- **Capabilities are code-defined**, not DB rows: `app/capabilities/definitions.py`
  (id, schemas, `supported_media_types`, `cost_class`, `enabled`), exposed
  read-only via `GET /capabilities`. `app/capabilities/registry.py` maps each id
  to a handler; anything unmapped falls back to `NotImplementedHandler` (job
  succeeds with `{"status":"not_implemented"}`).
- **One uniform handler interface**: `CapabilityHandler.run(ctx: JobContext) ->
  HandlerResult` (`app/capabilities/context.py`). Three families:
  - `ProviderBackedHandler` — AI ops; routes to the configured inference
    provider, returns JSON.
  - `LocalToolHandler` — deterministic ops; reads `ctx.input_path` /
    `ctx.input_paths` + `ctx.params`, writes files into `ctx.work_dir`, returns
    `OutputFile`s. Subprocess logic lives in `app/tools/{runner,ffmpeg,imagemagick}.py`.
  - `NotImplementedHandler` — fallback.
- **`execute_job` (`app/services/jobs.py`) is the orchestration hub**: builds the
  `JobContext` (resolving the primary `asset_id` plus any `input.asset_ids` to
  file paths + a temp work dir), runs the handler, and **persists each
  `OutputFile` as a derived Asset** via `assets.create_derived_asset`, then sets
  `job.output.outputs[] = {asset_id, filename, media_type, size_bytes, download_path}`.

To **add a capability**: add a `CapabilityDef` to `definitions.py`, write a
handler (subclass the right base), register it in `registry.py`. For a local
tool, put the subprocess call in `app/tools/`.

### Provider abstraction (AI)

Selected **only** in `app/providers/factory.py` by `INFERENCE_PROVIDER`
(`mock`|`http`) — routes and job-creation never know the provider. Swapping to a
real hosted endpoint is env-only (`INFERENCE_PROVIDER=http`, `INFERENCE_BASE_URL`,
`INFERENCE_API_KEY`, per-capability path vars). Add a provider by subclassing
`BaseInferenceProvider` and registering it in the factory.

### Assets, storage, derived outputs

- `Asset.source_asset_id` (self-FK) records provenance: `null` = uploaded
  original, set = derived from a job. This makes operations **composable** — one
  job's output `asset_id` feeds the next job as input (e.g. `*.concat` via
  `input.asset_ids`). This composability is the foundation for a future agentic
  planner that uses capabilities as tools.
- `app/services/storage.py` `StorageService` (local impl) is intentionally narrow
  (`save_upload`/`get_path`/`open_for_read`/`delete_asset`/`calculate_checksum`)
  so S3/MinIO can replace it. Layout: `projects/{pid}/assets/{aid}/original/{name}`.
- Uploads validate extension (415) and `MAX_UPLOAD_SIZE_MB` (413); derived
  outputs are trusted and skip those gates. Projects/assets use **soft deletes**
  (`deleted_at`); queries filter `deleted_at IS NULL`.

### Layering convention

`api/v1/` (thin routes, typed Pydantic in/out) → `services/` (business logic, all
DB access) → `models/` (SQLAlchemy) / `schemas/` (Pydantic). Domain errors are
`AppError` subclasses in `app/core/errors.py`, mapped to HTTP status by a single
handler in `app/main.py` — raise these from services rather than HTTPException.
List endpoints that can grow return a generic `Page[T]` envelope
(`{items, total, limit, offset}`); see `GET /projects/{id}/jobs`.
