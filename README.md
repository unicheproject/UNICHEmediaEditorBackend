# UNICHE Media Editor — Backend (MVP)

Backend + infrastructure foundation for a browser-based Media Editor for
cultural-heritage professionals.

The architecture follows the target design: an **internal API** between the UI
and the backend toolbox, a **Capability Registry** of operations, an **async
Job Runner** for long-running media/AI work, an **Asset Store** for uploads and
metadata, and AI-assisted operations routed to **hosted inference** behind a
provider abstraction.

The stack runs **fully offline by default** (mock inference, no API keys) while
proving the real async-job + hosted-inference plumbing. Switching to a real
hosted endpoint is an environment-variable change — no route or job-creation
code changes.

## Tech stack

- **Python 3.12**, **FastAPI**, **Pydantic v2**
- **PostgreSQL** + **SQLAlchemy 2.x async** (asyncpg) + **Alembic**
- **arq** worker (Redis-backed) for the async Job Runner
- **httpx** for hosted inference calls
- Local-filesystem asset storage behind a swappable `StorageService`
- **pytest**, **ruff**, **mypy**
- **Docker Compose** for the full stack

### Why arq for the job runner?

`arq` is async-native and Redis-backed. Because the API, ORM (SQLAlchemy async)
and HTTP client (httpx) are all async, `arq` lets job handlers `await` directly
with no sync/async bridging — unlike Celery/Dramatiq/RQ, which are sync-first.
It is lightweight, runs as a second container off the same codebase, and shares
the Redis instance. For an async FastAPI MVP it is the lowest-friction choice.

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Then in another shell, apply migrations (first run only):

```bash
docker compose exec api alembic upgrade head
```

- API docs (Swagger): http://localhost:8000/docs
- OpenAPI JSON: http://localhost:8000/openapi.json
- Health: http://localhost:8000/health
- Placeholder page: http://localhost:8000/

Production CI/CD and Hetzner VM setup are documented in
[DEPLOYMENT.md](DEPLOYMENT.md). The regular `docker-compose.yml` and optional
override remain development-only.

The `api` container runs uvicorn with `--reload`; the `worker` container runs
`arq app.workers.worker.WorkerSettings`. Both share the codebase and the
`storage` volume. Postgres data persists in the `pgdata` volume.

## Media subprocess execution

CLI tools use argv (no shell), closed stdin, a private temporary working
directory and an explicit system-path/UTF-8/temp environment without inherited
application credentials. Captured query stdout is capped at 1 MiB; ordinary
processing stdout is discarded. Only the last 16 KiB of stderr is retained,
with diagnostics at server DEBUG level, never in public tool errors. Timeout
or cancellation terminates the process group (TERM, then KILL after 250 ms)
and bounds child reaping to one second. Existing operation timeouts remain;
metadata/frame probes and waveform extraction keep their bounded streaming I/O.

FFmpeg picks a demuxer from a file's contents, not its name. An uploaded
`.mp4` that is really an HLS playlist used to be opened as HLS, and FFmpeg read
the local media it pointed to into the output. Every FFmpeg/FFprobe input
opened from a file now carries `-protocol_whitelist file -format_whitelist
mov,matroska,avi,mp3,wav,flac,ogg,png_pipe,jpeg_pipe,webp_pipe,tiff_pipe`
(`app/tools/demuxers.py`). These are the FFmpeg 5.1 demuxers of every accepted
format; none follows references to other files (mov external tracks are off by
default). Playlists, concat and `image2` filename patterns are not allowed.
FFmpeg still picks the demuxer for the contents and then refuses it if it is
not listed. Only application-built inputs are exempt, through a forced format:
lavfi sources, the concat lists of the application's own intermediate files,
and subtitles validated before use. `media_process` refuses any FFmpeg/FFprobe
command with an input that lacks this, so a new call site cannot omit it. Each
input's options must contain both settings exactly once with these values
(FFmpeg uses the last of repeated options); a forced format is exempt only
alone and only for those trusted inputs. Agent reuse identities (job nodes
and fused render groups) include `media_input_policy`, so results computed
before this policy are recomputed once instead of reused.
Shot detection (OpenCV, which ignores these options) first opens the same file
with a restricted FFprobe. Uploads are unchanged: such a file gets unavailable
metadata, and every Job on it fails without an Asset.

This is not a complete sandbox: native thread cancellation, resource quotas
and deployment isolation remain separate. The libass `subtitles` filter opens
its file itself, which is safe only because subtitles are validated first.

## Authentication & authorization

The API is a Keycloak resource server; every `/api/v1` route requires a valid bearer token
(`GET /health` is public). Authorization is delegated to the UNICHE Catalogue — the backend forwards
the user's token on outbound calls and keeps **companion project rows keyed by the catalogue's
project UUID**. Relevant env (see `.env.example`):

```
IDP_ISSUER_URI=https://idp.uniche-eccch.eu/realms/uniche   # MUST match the browser-visible issuer
REQUIRED_AUDIENCE=uniche-platform
CATALOGUE_BASE_URL=http://uniche-catalogue.localhost:8080   # no /api/v1 suffix
TOOL_SLUG=media-editor
```

No Keycloak client is needed for Phase 1 (the resource server only validates tokens; the deferred
reconcile sweep is what would need a service account). See `../plans/media-editor-auth-and-sync.md`.

## Running migrations

```bash
# Apply all migrations
docker compose exec api alembic upgrade head

# Create a new autogenerated migration after changing models
docker compose exec api alembic revision --autogenerate -m "describe change"

# Roll back one revision
docker compose exec api alembic downgrade -1
```

## Asset technical metadata

New uploads are saved and returned immediately with
`technical_metadata_status: "pending"`; the ARQ worker probes them
asynchronously. Job-produced media is probed in the existing worker before
the derived Asset is committed. Probe failures leave the Asset available with
`technical_metadata_status: "unavailable"` and nullable technical fields.
Asset reads include the normalized summary and bounded stream details.

After applying the metadata migration, backfill existing non-deleted Assets
explicitly (the migration does not read media files):

```bash
docker compose exec api python -m app.scripts.backfill_asset_metadata
```

The command can be rerun safely; already available Assets are skipped. An
optional `--project-id UUID` scopes the operation and `--batch-size N` controls
the number of Asset IDs loaded at a time.

`width`/`height` are the stored (coded) dimensions; `display_width` and
`display_height` are the size as shown, after video display rotation or EXIF
orientation, and are what crop coordinates refer to. Migration
`0016_asset_display_size` leaves them null for already-probed Assets, because
only a fresh probe can tell whether a file is rotated. Fill them with:

```bash
docker compose exec api python -m app.scripts.backfill_asset_display_size
```

It writes only the two display fields, is safe to rerun, and accepts the same
`--project-id` and `--batch-size` options.

## Asset waveform previews

Audio Assets and video Assets containing audio receive best-effort waveform
enrichment. Uploads use the existing asynchronous ARQ enrichment task; derived
outputs are enriched in their worker after persistence. Preview failures do not
fail uploads or media Jobs. Images/subtitles are unsupported; silent video is
`unavailable` with reason `no_audio`.

`GET /api/v1/assets/{id}/waveform` uses owning-project authorization and returns
version-1 JSON: `status`, `version`, `channels`, `sample_rate`, `bucket_samples`,
`bucket_count`, `duration_seconds`, and `peaks`. The first audio stream is
downmixed to mono at 8000 Hz. At most 1024 chronological min/max pairs are
normalized to [-1,1] from signed 16-bit PCM (not normalized to each Asset's
maximum amplitude). Each pair covers `bucket_samples / sample_rate` seconds;
only the final bucket may be shorter. Streaming reduction, a 120-second decoder
timeout, and a 64-KiB sidecar limit bound resource use and artifact size.

Pending previews return HTTP 202 with `Retry-After: 5`; unavailable previews
return HTTP 200 with `status: "unavailable"` and a stable `reason_code`, without
peaks. Asset reads expose independent `waveform_status`/`waveform_reason` fields.
Deleted/inaccessible Assets remain concealed (404); physically expired Temp
Assets return 410. Retrieval never decodes media. Private checksum/version-bound
sidecars are cached inside the Asset directory, reused across renames, and
removed by existing Temp cleanup. Missing/corrupt sidecars are reported as
unavailable rather than returning invented data.

After applying migration 0013, explicitly backfill old Assets or repair missing
previews (no automatic startup scan):

```bash
docker compose exec api python -m app.scripts.backfill_asset_waveforms --batch-size 100
```

Optional `--project-id UUID` limits scope. Batches must be 1–1000. Reruns are
safe: valid cached previews skip decoding; failed/missing previews are retried.

## Video splitting

`video.split` divides the complete first video stream at `markers` (timestamps
in seconds), producing N+1 ordered segment Assets. Markers must be finite,
positive, distinct, strictly increasing and before the measured video end;
every interval must contain frames. Invalid static values return HTTP 422;
duration/frame-dependent invalidity fails the Job without publishing Assets.

Cuts snap to the first source frame at or after each marker, not to existing
keyframes. Explicit interval re-encoding preserves each frame's ownership and
uses the canonical container-compatible codec choices (H.264/AAC for MP4,
VP9/Opus for WebM; MOV/MKV retain their supported container). The first source
soundtrack is trimmed to the same snapped intervals; silent input remains
silent. No multiple-track or stream-copy mode is provided.

All segments are probed before publication: frame counts must agree exactly,
and video duration tolerance is one measured source frame (the longest
observed frame interval for variable-rate video), plus 2 ms for container
timestamp quantization. Combined tolerance accumulates per segment. Audio
encoder delay is not included in video duration. Frame probing streams bounded
records with the existing 60-second timeout; unknown/ambiguous final-frame
duration fails cleanly instead of guessing. Splitting re-decodes each interval
and is not a lossless operation.

## Running tests

Tests use an isolated SQLite database and run jobs eagerly (no Redis/arq
needed), with `INFERENCE_PROVIDER=mock`.

```bash
# In the api container
docker compose exec api pytest

# Or locally
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy app
```

## End-to-end example (curl)

```bash
BASE=http://localhost:8000/api/v1

# 1. Create a project
PROJECT=$(curl -s -X POST $BASE/projects \
  -H 'Content-Type: application/json' \
  -d '{"name":"Heritage Reel","description":"1950s archive"}')
PROJECT_ID=$(echo "$PROJECT" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# 2. Upload an image (or audio/video)
ASSET=$(curl -s -X POST $BASE/projects/$PROJECT_ID/assets \
  -F "file=@/path/to/photo.jpg")
ASSET_ID=$(echo "$ASSET" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# 3. List assets in the project
curl -s $BASE/projects/$PROJECT_ID/assets

# 4. List available capabilities
curl -s $BASE/capabilities

# 5. Create an image.caption job
JOB=$(curl -s -X POST $BASE/jobs \
  -H 'Content-Type: application/json' \
  -d "{\"capability_id\":\"image.caption\",\"project_id\":\"$PROJECT_ID\",\"asset_id\":\"$ASSET_ID\"}")
JOB_ID=$(echo "$JOB" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# 6. Check job status + output (queued -> running -> succeeded)
curl -s $BASE/jobs/$JOB_ID

# 6b. (Optional) Stream job progress via Server-Sent Events
curl -N $BASE/jobs/$JOB_ID/events

# 7. Same flow for audio.transcribe (upload an .mp3/.wav first)
curl -s -X POST $BASE/jobs \
  -H 'Content-Type: application/json' \
  -d "{\"capability_id\":\"audio.transcribe\",\"asset_id\":\"<AUDIO_ASSET_ID>\"}"
# Optional spoken-language hint (ISO-639-1, as for subtitle.autogenerate); omit to auto-detect
#   ... ,\"input\":{\"language\":\"el\"}
```

## API surface

| Area | Method & path |
|------|---------------|
| System | `GET /health` |
| Projects | `POST /api/v1/projects` · `GET /api/v1/projects` · `GET/PATCH/DELETE /api/v1/projects/{id}` |
| Assets | `POST /api/v1/projects/{id}/assets` · `GET /api/v1/projects/{id}/assets` · `GET /api/v1/projects/{id}/assets/search` (filtered page) · `POST /api/v1/projects/{id}/assets/bulk-delete` · `GET /api/v1/assets/{id}` · `PATCH /api/v1/assets/{id}` (rename; 409 if the name is taken) · `POST /api/v1/assets/{id}/duplicate` · `GET /api/v1/assets/{id}/download` · `GET /api/v1/assets/{id}/thumbnail` (JPEG image/video preview) · `GET /api/v1/assets/{id}/waveform` (bounded JSON audio preview) · `DELETE /api/v1/assets/{id}` |
| Capabilities | `GET /api/v1/capabilities` · `GET /api/v1/capabilities/{id}` |
| Jobs | `POST /api/v1/jobs` · `GET /api/v1/jobs/{id}` · `GET /api/v1/projects/{id}/jobs` (paginated) · `GET /api/v1/jobs/{id}/events` (SSE) |
| Agent | `POST /api/v1/agent/sessions` · `GET /api/v1/agent/sessions/{id}` · `POST /api/v1/agent/sessions/{id}/messages` · `GET /api/v1/agent/plans/{id}` · `POST /api/v1/agent/plans/{id}/approve` |

Asset names are unique per project (case-insensitive) among live Assets: uploads,
Job outputs and duplicates get `_2`, `_3` before the extension, and a rename to a
taken name returns 409. Job outputs carry their defining parameters in the name
(`clip_trim_10-240.mp4`, `scan_resize_1600x1067.jpg`). Assignment runs under a
per-project row lock on PostgreSQL.

`GET /api/v1/projects/{id}/jobs` is paginated: `?limit=` (default 50, max 200)
and `?offset=`, newest first. An optional `?status=queued|running|succeeded|failed|cancelled`
filter applies before pagination and total calculation. It returns a page envelope:
`{ "items": [...], "total": N, "limit": 50, "offset": 0 }`.

`GET /api/v1/projects/{id}/assets/search` requires the same project authorization
and returns `{ "items": [AssetRead, ...], "total": N, "limit": 50, "offset": 0 }`.
Optional `q` matches a literal, case-insensitive substring of the display name
(`original_filename`, including renames); `%`/`_` are not wildcards. Outer
whitespace is ignored; empty text means no name filter. `q` is at most 512
characters and cannot contain NUL. Optional `media_type` is
`image|audio|video|subtitle|unknown`; omit it for all types.
`sort_by=created_at|modified_at|name|size|type` defaults to `created_at`, and
`sort_order=asc|desc` defaults to `desc`. `created_at` is the date added;
`modified_at` is the last user change (currently rename) and equals `created_at`
until then; unlike `updated_at` it ignores background analysis such as metadata
probing and waveforms. Name sorting is case-insensitive,
size is in bytes, and type is alphabetical; Asset ID breaks ties in the same
direction. Pagination uses `limit` (default 50, range 1–200) and `offset`
(default 0, range 0–2147483647); `total` counts filtered matches before pagination.
Deleted Assets are always excluded; `include_intermediate=true` includes active
intermediates (default false). Invalid/unknown query parameters return 422.
This is a read-only query: it does not probe media or enqueue work. A small
`q`/`limit` query can also supply autocomplete candidates. Ordering is stable
for unchanged rows, not a snapshot across concurrent uploads/deletes.
The existing `/projects/{id}/assets` endpoint still returns its unpaginated array.

`GET /api/v1/projects/{id}/assets` excludes Temp Processed Assets
(`is_intermediate=true`) by default. Pass `?include_intermediate=true` to
include them. Asset responses expose `temp_expires_at` and
`physical_expired_at`; the latter is populated after automatic expiry removes
the media binary while retaining the Asset and job/Agent audit metadata.
Download and thumbnail requests for an expired binary return HTTP 410.
Intermediate checkpoints expire after 72 hours; final Agent outputs and
uploaded originals are not part of this retention cleanup.

Project and asset deletes are **soft** (`deleted_at`). Uploads validate file
extension (415 if unsupported) and size against `MAX_UPLOAD_SIZE_MB` (413 if
too large).

`POST /api/v1/projects/{id}/assets/bulk-delete` requires project authorization
and JSON `{ "asset_ids": ["ASSET_UUID", "OTHER_ASSET_UUID"] }` with 1–100
unique UUIDs (invalid bodies return 422). HTTP 200 returns
`{ "results": [{ "asset_id": "ASSET_UUID", "status": "deleted" }, ...] }`
in request order. `not_found` means no active matching Asset in this project;
missing, already deleted and other-project IDs are deliberately indistinguishable.
Matching Assets are soft-deleted together in one transaction, even when other
IDs do not match. A database failure rolls back the whole batch and returns a
controlled 500 rather than per-item success. Repeating a completed request
returns `not_found` without changing deletion timestamps. Binaries, provenance,
Job/Agent history and saved composition references are retained, as for single
deletion; this endpoint does not reclaim disk space or cancel Jobs.

Supported uploads: images `jpg jpeg png webp tiff`, audio `mp3 wav m4a flac
ogg`, video `mp4 mov webm mkv`.

## Architecture notes

```
UI ──▶ FastAPI routes (app/api/v1) ──▶ services (app/services)
                                          │
                 enqueue job id ──────────┤──▶ Postgres (projects/assets/jobs)
                       │                   └──▶ StorageService (local FS, S3-ready)
                       ▼
                 Redis (arq queue)
                       │
                       ▼
        arq worker (app/workers) ──▶ services.jobs.execute_job
                                          │
                              capability registry ──▶ handler
                                                        │
                                          provider factory ──▶ Mock | HTTP provider
```

Key boundaries:

- **Routes never execute work and never know the provider.** `POST /jobs`
  validates references, persists a `queued` job, and enqueues `run_job(job_id)`.
  A caught, unconfirmed dispatch failure conditionally changes only `queued`
  to `failed`, records a fixed dispatch error, and returns the existing HTTP
  500 error envelope. Confirmed runnable tasks or a worker/terminal state that
  wins the race retain the normal 201 response. Queue cleanup errors cannot
  invalidate acknowledged dispatch. Process loss between commit and dispatch,
  or a failed compensation commit, still requires separate recovery; this is
  not crash-safe or exactly-once dispatch.
- **The worker** atomically claims only `queued` Jobs. Delivery of a running or
  terminal Job is a no-op; a Job ID has one execution attempt and is never
  automatically reclaimed. Media/provider execution holds no Job lock.
  Cancellation and failure use conditional transitions, so stale workers cannot
  overwrite terminal state. Outputs are copied/probed without Asset rows, then
  one short conditional transaction publishes all output Assets, `Job.output`,
  and `succeeded` together. Cancellation winning first prevents publication;
  publication winning first prevents later cancellation. Agent physical Jobs,
  render groups, and `project.export` use the same fences. Waveform enrichment
  runs best-effort after publication. Unpublished objects are removed best-effort;
  abrupt process death or an unverifiable DB failure can leave unreferenced
  storage. This is at-most-one authoritative publication, not exactly-once
  provider invocation.
- **Abandoned work is recovered, not retried** (`app/services/reconciliation.py`).
  A worker killed outright (SIGKILL, OOM, crash, host restart) used to leave its
  Job `running` forever: arq redelivers the task, but the claim only accepts
  `queued`. arq cancels every task after the worker's job timeout
  (`WORKER_JOB_TIMEOUT_SECONDS`, 1800 s), so a Job or Agent attempt started
  more than that plus 190 s ago (arq's 10 s in-progress margin and a 180 s
  grace) cannot belong to a live worker. Such Jobs become `failed` ("The worker
  stopped before this job finished. Please run it again."). Such Agent plans
  become `failed` with their attempt, the active attempt is cleared, and they
  can be approved again; completed steps stay reusable. A Job still `queued`
  10 minutes after creation fails with the dispatch message only when arq
  definitely no longer holds its task; an unreachable or uncertain queue leaves
  it alone. Progress inactivity is never used. Recovery runs at worker startup
  and every 5 minutes (arq cron, one run per time), in short conditional
  updates. Workers publish, report progress and finish only while their Job is
  still `running`, and an Agent executor writes plan/attempt state only while
  its attempt is still the plan's active, running attempt. So a late worker or
  executor cannot overwrite a recovery, a newer attempt, or each other.
  Running work counts as abandoned 1990 s after `started_at`, and recovery
  runs every 5 minutes, so under normal conditions it is recovered about 33–38
  minutes after it started. It can take longer if the worker, its schedule,
  the database or the queue is delayed or unavailable; this is not a
  guaranteed upper bound. Nothing is retried or requeued automatically.
  Recovery deletes nothing: unreferenced
  storage from a crash and `/tmp` work directories remain, as do approved plans
  whose task was never queued.
- **Capabilities are code-defined** in `app/capabilities/definitions.py` and
  exposed read-only via the API — they are not database rows.
- Capability library records include canonical `layer` values
  (`deterministic_media`, `ai_assisted`, `composition`, or
  `agent_orchestration`): local predictable media operations, model-assisted
  operations, media assembly/rendering, and conversational planning/orchestration,
  respectively. The current operation registry does not contain a separate
  Agent capability record; Agent is an API surface over these same operations.
  Each record also has a product-level `tool` label. Operations backed by a
  swappable provider say `Configured inference backend`; definitions without
  a working handler say `Not implemented`. `GET /capabilities` remains
  enabled-only by default; `?include_disabled=true` additionally lists
  code-defined disabled operations, which remain non-executable. `enabled`,
  readiness/availability, and input applicability are separate dimensions.
  Discovery does not enable execution, and this API has no institution-level
  capability mutation or provider-configuration endpoints.
- Capability `input_schema` values are the shared execution/discovery
  contract: finite choices use JSON Schema `enum`; dependent finite choices
  may add property-level `x-option-constraints` with `depends_on` and an
  `allowed_values` map. Standard JSON Schema conditionals enforce the same
  combinations at runtime. For `video.transcode`, codec fields are optional
  (omission keeps FFmpeg's container default), and the advertised option map
  describes which explicit codecs are supported for each output container.
- `video.transcode.quality` optionally selects `high`, `balanced`, or
  `compressed`, prioritizing visual quality, a balance, or smaller files.
  Selecting it requires an explicit supported `video_codec`. Omitting it keeps
  the previous FFmpeg command and encoder defaults. The codec-specific CRFs
  are respectively H.264 18/22/28, HEVC 22/28/34, and VP9 24/32/40; VP9 also
  uses `-b:v 0` for unconstrained constant quality. `compressed` additionally
  caps the video bitrate at 75% of the probed source video bitrate when known
  (`-maxrate`/`-bufsize` for H.264/HEVC, constrained `-b:v` for VP9) and reports
  it as `max_video_bit_rate`; audio and container overhead are not capped, so
  a smaller total file is likely but not guaranteed. These scales do not imply
  equivalent quality across codecs. Audio settings, dimensions and frame rate
  are unchanged. File size depends on the content; there is no target-filesize
  guarantee. Discovery and Agent grounding expose the same optional enum,
  explanations and conditional codec requirement used by Job admission.
- `audio.transcode` offers the maintained audio Asset families: `mp3`
  (`libmp3lame`), `wav` (`pcm_s16le`), `m4a` (`aac`), `flac` (`flac`), and
  `ogg` (`libvorbis` or `libopus`). Codec omission keeps FFmpeg's container
  default (Vorbis for Ogg). Discovery exposes the same `enum`,
  `x-enum-descriptions`, `x-option-constraints` and JSON Schema conditionals
  used by admission and Agent grounding; arbitrary FFmpeg encoder names
  are not supported capability choices.
- `image.colour.adjust` uses integer percentages: brightness/saturation
  default to 100 (unchanged), with minimum 0 (black/grayscale); contrast is
  -100..100 with default 0 (unchanged). Brightness/saturation have a deliberate
  product maximum of 200, independently of ImageMagick's technical limits.
- Capability `applicability` is the generic input-scope contract serialized by
  `GET /capabilities`: `primary_asset` is `required`, `optional`, or
  `forbidden`; `project_context` is `required` (the caller must name the
  project) or `derived` (the project may be inferred from a primary or nested
  Asset reference). `supported_media_types` describes only an accepted
  top-level primary Asset and is empty when primary Assets are forbidden.
  Parameter Asset references use `x-media-asset: {role, media_types}` on the
  UUID schema node. For arrays, read `items.x-media-asset`: its `media_types`
  constrains every referenced Asset, independently of primary selection.
  Array-level `x-media-asset` remains as a compatibility copy of the same
  declaration; `minItems` and ordering are unchanged. Scalar references carry
  the same metadata directly on their property. Job and Agent validation use
  these canonical declarations and authoritative Asset records. Job admission enforces
  this contract both at the API boundary and in the Job service. This is
  structural applicability, not institutional readiness/policy; `enabled`
  remains the legacy code-owned global switch.
- **ME-16 readiness** is separate from applicability. Every definition declares
  implementation `available` or `unavailable` and a deployment requirement.
  Capability responses add `readiness: {implementation, deployment, reason_code}`
  and `availability: {executable, reason_code}`. Deployment is `ready`,
  `configuration_required`, or `unavailable`. Executability requires the global
  switch, an available implementation, and ready deployment configuration.
  Institutional policy is not included in Phase 1. Clients use applicability for
  input selection and `availability.executable` for execution, never `enabled`
  alone. Definitions with no implementation remain listed with `not_implemented`.
  Job service admission rejects unavailable operations with HTTP 422
  `capability_unavailable` before persistence/queueing; workers recheck readiness
  and fail old queued Jobs truthfully. Agent catalogs include executable
  operations only; plan validation, approval, and execution enforce readiness.
  Registry startup checks prevent declaration/handler drift.
- Provider readiness performs configuration checks only, with no live health
  checks. Mock inference is for provider-level development/tests and is never
  publicly executable. HTTP supports caption/transcription when its base URL and
  operation path are configured; OpenRouter supports TTS/transcription when its
  key, base URL and model are configured. Other combinations return
  `provider_not_configured`. Transient failures remain ordinary failed Jobs and
  do not change capability readiness. Credentials are never exposed in readiness.
- `image.upscale` has an available CUDA-only implementation. The API has no worker
  GPU device, so it uses the explicit deployment declaration
  `IMAGE_UPSCALE_WORKER_READY` (default false: `configuration_required`). Set it to
  true in API and worker only after verifying CUDA and the vendored model on all
  workers that receive Jobs. The worker checks CUDA/model locally before work;
  missing CUDA fails with `gpu_unavailable`, missing model/runtime with
  `configuration_required`. Listing never loads a model or performs inference.
  There is no CPU fallback. `video.upscale` remains implementation-unavailable
  with `not_implemented`.
- Saved project compositions use immutable `CompositionRevision` documents.
  Video clips may carry optional source in/out points (`source_start`,
  `source_end`: seconds in the source Asset, not timeline positions), with
  `video.trim`'s rules. They are checked on save when the
  source metadata is known, and always again when the clip is normalized,
  which also trims it, so there is no extra encode. Unset values are not
  serialized, so untimed clips and existing revision hashes are unchanged.
  Still images join the same ordered track as `{"kind": "image", "asset_id",
  "duration"}` (1/30–600 s, at most 100 megapixels). A clip without `kind` is a
  video clip; `kind: "video"` is accepted but never stored. Each image becomes
  one normalized segment in a single encode (read at 1 fps; the fps filter
  repeats frames), with silent audio, the render policy's codecs, orientation
  from ImageMagick's header probe (FFmpeg ignores, e.g., the TIFF tag) and
  transparency composited onto black.
  An optional second track `{"kind": "audio", "clips": [...]}` (1–64 clips)
  places audio Assets at `timeline_start` seconds on the output, with optional
  `source_start`/`source_end` and `volume` 0..1 (default 1). One audio-only
  pass mixes them over the programme: the bed's mix/duck rules are unchanged.
  For replace, the bed is padded to the video's length rather than shortening
  it. Clips are summed (overlaps allowed, no normalisation), trimmed at the
  video's end, and a clip starting at or after the end fails the render (or
  the save, when every length is known). Without an audio track nothing
  changes. Export snapshots add `ordered_audio_clip_sources` only when present.
  `app/composition/render.py` compiles one revision into a typed semantic
  render request; `app/services/composition_render.py` revalidates and resolves
  its project-owned Assets; `app/tools/compose_renderer.py` delegates to the
  same FFmpeg compose implementation used by direct `video.compose` Jobs.
  Composition semantics (`ComposeRenderRequest`) remain separate from the
  trusted encoding policy (`ComposeRenderPolicy`). Direct `video.compose`
  defaults to its existing MP4/H.264/AAC behavior; the internal lossless
  policy carries Matroska/FFV1/FLAC through normalized segments, stream-copy
  concat, audio-bed processing, and subtitle burn-in. Segment normalization
  preserves the first source soundtrack, normalizing to 44.1 kHz stereo;
  audio is padded/trimmed to the first video's duration, and clips without
  audio receive silence. Ordered soundtracks follow the video timeline.
  The optional bed retains existing `mix`, `duck` (bed volume capped at 0.2),
  and `replace` semantics; replacement beds remain unscaled. In `replace` the
  video keeps its full length: a shorter replacement is padded with silence and
  a longer one trimmed at the video's end (it used to cut the video at the end
  of a shorter bed). Public `audio.mix` instead applies `music_volume` to supplied
  audio in `mix` and `replace` (including zero/mute); `duck` caps it at 0.2. Its
  `replace` on a video also keeps the video's length; on audio-only media the
  replacement is the whole result, as before. Embedded cover art (an
  attached picture in an MP3/M4A, or in a video file) is not treated as
  video: it is neither measured nor copied, so a song with a cover gets the
  full replacement and a video keeps all its real frames. The same holds for
  `mix` and `duck`: a cover is never copied (it used to end the result after
  about 0.1 s), and a video without its own sound keeps its full length, with
  the music (at its mix or duck volume) padded with silence or trimmed to the
  video instead of cutting it.
  Lossless policy uses FLAC rather than a lossy audio intermediate; channel/rate
  normalization is still applied, not bit-identical source stream copying.
  Archival
  `project.export` selects that policy from the start without changing saved
  compositions. `project.export` is a normal asynchronous Job bound to the
  exact submitted revision; its immutable Job input snapshots the revision hash,
  compiler/preset versions, ordered source checksums, and fully resolved typed
  settings. Web, Social, and Preview render the composition with the unchanged
  default policy and apply one bounded H.264/AAC delivery transform. Archival
  renders directly to MKV/FFV1/FLAC with no lossy composition intermediate.
  The final result is one ordinary visible derived Asset; renderer-local files
  remain inside the Job work directory.
- Project templates (MED/028) are built-in, versioned JSON files in
  `app/composition/templates/<id>/vN.json`, listed explicitly in
  `TEMPLATE_FILES` and validated at startup. A template is a composition whose
  clips name a *slot* (image, video or audio; optional; `multiple` up to 12)
  instead of an Asset. `GET /composition-templates[/{id}[/versions/{n}]]`
  lists and reads them. `POST /projects/{id}/composition/template` takes
  `{template_id, version, bindings}`, compiles an ordinary composition (clip
  and track IDs are UUIDv5 names, so the same bindings give the same
  document) and saves it through the normal PUT path, with the same
  preconditions, statuses and response. Applying runs no Jobs and creates no
  Assets or projects; the template is not recorded in the revision.
  Starting a project from a template therefore takes three steps: create the
  project, upload or create its Assets, then apply. There is no single
  create-from-template call.
  Published versions are immutable (their hashes are pinned in tests):
  changes go in a new version file.
- `video.subtitle.embed` (SRT/VTT) adds a selectable track (`mode: "soft"`,
  the default) or burns subtitles into the picture (`mode: "burn"`, MP4/H.264).
  Burn mode takes optional styling (MED/030); every field is optional:
  `font` (`dejavu-sans`, `dejavu-sans-bold`, `dejavu-serif`,
  `dejavu-serif-bold`), `font_size` (12–96), `text_colour` and
  `outline_colour` (`#RRGGBB`), `outline` (0–4), `position` (`bottom`/`top`)
  and `margin` (0–200). Sizes and margins are libass units that scale with the
  video height (24 is about one twelfth of it). They become a libass
  `force_style` built only from these validated values; there is no raw
  style, font name or path. Without styling fields the burn command is
  exactly as before. Any styling field with `soft` (or no `mode`) is a 422.
  Example input: `{"subtitle_asset_id": "<id>", "mode": "burn", "font":
  "dejavu-sans-bold", "font_size": 28, "text_colour": "#FFFFFF",
  "outline_colour": "#000000", "outline": 2, "position": "bottom", "margin":
  20}`. Saved compositions and `project.export` do not render subtitles; burn
  them into a clip, or into the exported video. A burn always produces
  MP4/H.264, also after an Archival (FFV1) export.
- Subtitle files are checked when a Job uses them (`video.subtitle.embed` in
  both modes, and `video.compose` before any rendering), so Assets uploaded
  earlier are covered too; uploads themselves are unchanged
  (`app/tools/subtitles.py`). A file must be UTF-8 text (BOM allowed), at
  most 10 MiB, and SRT or WebVTT by its content, whatever its extension: SRT
  starts with a cue number and a timing line; WebVTT starts with `WEBVTT`.
  It needs at least one cue with text whose end is after its start;
  zero-length cues are allowed but do not count, and cues after the end of the
  video are allowed. Rejected: empty, binary or non-UTF-8 files (re-save
  UTF-16 or Windows-1253 files as UTF-8), plain text, ASS/SSA, WebVTT `STYLE`
  and `REGION` blocks (FFmpeg 5.1 drops the cues after them), invalid
  timestamps, cues that end before they start, and cues without text. In
  WebVTT, every block after the header must be a cue or a `NOTE`; any other
  block is rejected wherever it is, and a cue must not sit inside the header
  block. A `region:` cue setting is fine. The `-->` arrow may have no
  spaces around it, but fractions need three digits (`00:00:01.500`), because
  FFmpeg 5.1 reads `.5` as 5 ms. In SRT, a timing pair with a broken arrow
  (`->`, `–>`, a missing arrow) where a timing line belongs (the first line
  of a block, or the line after a cue number) is rejected: FFmpeg would show
  it as text of the previous cue. Times inside cue text are fine. SRT keeps
  1–3 millisecond digits, read as whole milliseconds as FFmpeg does (`,5` is
  5 ms). Parsing is linear in the file size. FFprobe
  then reads the file with that format forced (30 s limit) and must find at
  least one cue; soft mode also forces it. FFmpeg never guesses the format of
  an unchecked file: binary data could be taken for MPEG-TS and keep libass
  busy until the 600 s runner timeout. An invalid file fails the Job with a
  plain message and no Asset (not a 422 at submission).
- Title cards (`media.titlecard`) and lower thirds (`video.lowerthird`) take
  optional typography (MED/030). Both take `font`: `dejavu-sans`,
  `dejavu-sans-bold`, `dejavu-serif` or `dejavu-serif-bold`, each mapped to
  one installed DejaVu file; no font name or path is ever passed through.
  - Title card: `font_size` (12–400 px), `text_align`
    (`left`/`center`/`right`) and `vertical_align` (`top`/`middle`/`bottom`).
    Without `font_size` the text is sized to fill the card, as before. With
    it, the size is kept exactly, and the Job fails if a word is wider than
    the card or the text is taller than it.
  - Lower third: `text_size` (12–64, default 36) and `caption_size` (10–48,
    default 22) on the unchanged 640×160 banner. The caption moves down
    with a bigger name.
  - Text that would run past the banner, or a name that would overlap the
    caption, now fails the Job instead of being clipped. This applies to
    requests without styling too.
  - Text prints exactly as typed: `%` (for example `%w`), backslashes and a
    leading `@` used to be interpreted by ImageMagick. Real newlines still
    break lines.
  - Colours (`background`, `foreground`) must be `#RGB`, `#RRGGBB` or a
    colour name (letters only); expressions such as `rgb(...)` or `red[..]`
    get 422.
  - Limits: title text up to 500 characters, name 200, caption 300. A title
    card is at most 3840 px per side and 3840×2160 in area (portrait allowed).
  - Without new fields, ordinary text renders pixel-identically. Lower thirds
    are still not scaled to small videos.
  - Agent: styled lower-third steps run as normal Jobs, not fused, so no
    field is dropped. The fused render version and the `video.lowerthird`
    reuse version were bumped.
- Export presets are defined once in `app/composition/export_presets.py`.
  Web fits without upscaling inside 1920×1080 (CRF 22, AAC 128 kbps stereo);
  Social requires `square` (1080×1080) or `portrait` (1080×1920) and
  preserves aspect ratio with padding, enlarging smaller sources to fit the
  canvas (CRF 21, AAC 128 kbps); Preview fits
  without upscaling inside 1280×720 (CRF 28, AAC 96 kbps); Archival is locked
  lossless FFV1/FLAC; Advanced is constrained to MP4/MKV with H.264/AAC plus
  optional even maximum dimensions, CRF 18–35, and AAC bitrate 64–320 kbps.
  MP4 deliveries use faststart. No arbitrary FFmpeg options are accepted.
- **Provider selection happens in exactly one place**:
  `app/providers/factory.py`, driven by `INFERENCE_PROVIDER`.

### Capability handlers

There are three handler families behind one uniform interface
(`CapabilityHandler.run(ctx: JobContext) -> HandlerResult`):

1. **Provider-backed** (AI, JSON output) — route to the configured inference
   provider: `image.caption`, `audio.transcribe`.
2. **Deterministic local tools** (`cost_class=deterministic`, **file output**) —
   run FFmpeg / ImageMagick and register each produced file as a new **derived
   Asset**:

   | family | capabilities | tool |
   |--------|--------------|------|
   | video | `video.trim` `video.split` `video.concat` `video.transcode` `video.export` `video.mute` `video.crop` `video.resize` `video.thumbnail` | FFmpeg |
   | image | `image.resize` `image.crop` `image.format` `image.colour.adjust` | ImageMagick |
   | audio | `audio.trim` `audio.concat` `audio.gain` `audio.normalize` `audio.fade` `audio.denoise` `audio.transcode` | FFmpeg |
   | video audio | `video.audio.extract` | FFmpeg |

   `audio.gain`, `audio.normalize`, `audio.fade` and `audio.denoise` accept
   audio or video inputs. Audio inputs remain audio outputs; video inputs keep
   the video stream via stream copy and return video with the processed
   soundtrack. These transforms fail when the video has no audio stream.
   `video.audio.extract` instead returns the first soundtrack as AAC/M4A and
   fails for silent video. Video audio transforms map only the first video
   and first audio streams; subtitle/data and additional tracks are omitted.

3. **Not-implemented fallback** — definitions without implementations explicitly
   declare unavailable readiness. Admission rejects them, and
   `NotImplementedHandler` raises a controlled unavailable error if reached by
   legacy Jobs or a programming error. It never reports success or creates Assets.

**Video concatenation.** `video.concat` normalizes ordered clips to the first
resolved input's dimensions and output container, with scale/pad, square pixels
and 30 fps. MP4/MOV/MKV use H.264/AAC; WebM uses VP9/Opus. The first soundtrack
is retained, with stereo silence supplied only for silent inputs; AAC uses
44.1 kHz and Opus 48 kHz. Compatible normalized intermediates are concatenated
without another encode, and the final container/streams are probed before
output publication.

**Derived assets & chaining.** Local-tool jobs write outputs under
`job.output.outputs[] = {asset_id, filename, media_type, size_bytes,
download_path}`. Each is a first-class `Asset` (downloadable via
`GET /assets/{id}/download`) with `source_asset_id` retaining a best-effort input anchor — so one
operation's output `asset_id` can feed straight into the next operation as input
(`input.asset_ids` for multi-input ops like `*.concat`). This composability is
what a future agentic planner will use to chain capabilities as tools.

**Recorded Asset provenance.** Asset list/detail responses expose
`provenance_kind` (`original`, `derived`, `unknown`), `ai_provenance` (`none`,
`generated`, `modified`, `inherited`, `unknown`) and nullable `source_job_id`.
Original means introduced as source material; Derived means produced/materialized
through a recorded editor operation. Unknown preserves insufficient historical
evidence. A null `source_asset_id` does not imply Original.

AI provenance records this editor's processing lineage only: `none` does not
certify human/external authorship. Direct AI generation/modification is declared
once in the capability registry (`ai_content_effect`). Deterministic descendants
of any AI-involved input inherit that lineage, even if a later transform removes
the perceptual AI content. Otherwise uncertainty propagates before `none`.
Agent initiation alone is not AI involvement; Temp lifecycle is independent.

Uploads may optionally send multipart `derived_from_job_id` for a successful,
same-authorized-project Job whose canonical `client_materialization_media_types`
permits that upload type. Initially only `audio.transcribe` permits subtitle
materialization (existing SRT/VTT uploads). These are possibly edited client
derivatives, not byte-level attestations. Manual uploads remain Original/none;
client classification fields are rejected. Deleting a Job clears its reference
but retains classification. Migration backfills only non-null source anchors
as Derived; other historical kinds and all historical AI states remain Unknown.

**Same-plan Agent retry reuse.** On re-approval of a failed/cancelled
`AgentPlan`, a conservative, explicitly versioned allowlist may reuse a
successful, single-output physical Job from an earlier attempt of that same
plan and project. Fused video operations are reused as one complete render
group, never as individual logical steps. Reused nodes have `reused: true`, a
null `job_id`, source Job/attempt references, and the original output Asset ID
in the new attempt's step/group audit. The attempt's `job_ids` contains only
Jobs physically created by that attempt. Temp Asset retention is not extended;
expired, deleted, missing, incomplete, or mismatched outputs are misses. There
is no direct-Job or cross-plan cache in this phase.

**Audio uses FFmpeg** (not SoX) so every accepted format — including m4a/AAC —
works; the handler layer stays modular if a SoX backend is wanted later.

System binaries (`ffmpeg`, `imagemagick`) are installed in the Docker image (shared
by api + worker). Tool jobs only run where those binaries exist; tests that
execute them are skipped if the binaries are absent.

## Inference providers

Selected by env var; the same job-creation path works for both:

- `INFERENCE_PROVIDER=mock` (default) — `MockInferenceProvider` returns
  deterministic placeholders for provider-level development/tests. Capability
  readiness reports `provider_not_configured`; public Jobs are rejected.
- `INFERENCE_PROVIDER=http` — `HTTPInferenceProvider` POSTs the asset file
  (multipart) plus the job `input` JSON to a configured endpoint, with
  `Authorization: Bearer <INFERENCE_API_KEY>`. It stores the JSON response in
  `job.output`, and marks the job **failed** on non-2xx, timeout, or invalid
  JSON.

Transport success does not imply inference success. Provider responses reject
explicit failure markers (`status`: `error`/`failed`/`not_implemented`, boolean
`success: false`/`ok: false`, or non-empty `error`); status matching ignores case
and surrounding whitespace. Informational `message`/`detail`, unknown statuses,
and extra fields remain allowed. Caption results require a non-blank string
`caption`; transcripts require an explicitly supplied string `text`, including
`""` for silence. No language or timing fields are required. OpenRouter validates
transcription before normalizing it. TTS rejects empty PCM before WAV wrapping,
requires complete 16-bit samples (silence is valid), and probes the generated
audio with the existing timeout/output-bounded ffprobe before publication.
Invalid results become controlled failed Jobs, not successful empty outputs.
Provider error messages do not include raw response bodies or transport
diagnostics. HTTP response buffering still has no application-level size cap.

Relevant env vars:

```
INFERENCE_PROVIDER=mock          # mock | http | openrouter
INFERENCE_BASE_URL=
INFERENCE_API_KEY=
INFERENCE_TIMEOUT_SECONDS=120
INFERENCE_IMAGE_CAPTION_PATH=/image/caption
INFERENCE_AUDIO_TRANSCRIBE_PATH=/audio/transcribe
```

To switch a deployment to real inference: set `INFERENCE_PROVIDER=http`,
`INFERENCE_BASE_URL`, `INFERENCE_API_KEY`, and the capability paths, then
restart the `worker` (and `api`). No code changes.

## How to add a new capability

1. Add a `CapabilityDef` entry to `app/capabilities/definitions.py`
   (id, title, description, input/output JSON schema, supported media types,
   `cost_class`, `enabled`, explicit implementation readiness and deployment
   requirement). It is now listed by `GET /capabilities`.
2. If it should do real work, create a handler in
   `app/capabilities/handlers/` and register it in `_HANDLERS` in
   `app/capabilities/registry.py`:
   - **AI / hosted inference** → subclass `ProviderBackedHandler`.
   - **Deterministic local tool** → subclass `LocalToolHandler`, implement
     `process(ctx)`: read inputs from `ctx.input_path` / `ctx.input_paths` and
     params from `ctx.params`, write outputs into `ctx.work_dir`, and return a
     `HandlerResult` with `OutputFile` entries (the job service registers them as
     derived assets). Put the subprocess call in `app/tools/`.
   - **Custom logic** → subclass `CapabilityHandler` directly.
3. If the capability uses the HTTP provider, add its endpoint-path setting in
   `app/core/config.py` and map it in `HTTPInferenceProvider._paths`.

Declare implementation-unavailable until a real handler is registered. Registry
startup invariants reject mismatched declarations; unavailable definitions can
be discovered but cannot execute.

## Conversational agent

The agent turns a natural-language request into an ordered **plan** of capability
calls, asks for missing parameters, and — after you approve — executes the
sequence, chaining each step's output asset into the next.

It is **registry-driven**: the planner's tool catalog is built from
the executable subset of `registry.list_enabled()`, so adding a ready capability
extends what the agent can do — no agent code changes.

Flow:

```
POST /agent/sessions            {project_id, asset_ids}      -> session (scope)
POST /agent/sessions/{id}/messages {content}                 -> {type: clarification|plan}
POST /agent/plans/{id}/approve                                -> enqueues execution
GET  /agent/plans/{id}                                        -> status + per-step jobs + result_asset_ids
GET  /agent/sessions/{id}                                     -> transcript + scope
```

A plan step references inputs by uploaded `asset_id` or by `@stepN` (a prior
step's output). The executor reuses the normal Job path, so every step is a
tracked Job producing a derived asset. The final asset id(s) land in the plan's
`result_asset_ids` (download via `GET /assets/{id}/download`).

**Planner backends** (env `AGENT_PROVIDER`):
- `mock` (default) — deterministic, rule-based; works offline and in tests.
- `openrouter` — plans via an OpenRouter LLM (`OPENROUTER_API_KEY`,
  `OPENROUTER_MODEL`, …), preserving reasoning across clarification turns.

Proposed plans are validated against the registry (capability exists, params
satisfy the capability `input_schema`, asset refs resolve, media types match)
before being shown; with an LLM backend, validation errors trigger a bounded
repair retry. API routes and job creation never know which backend planned.

## How to add a new inference provider

1. Subclass `BaseInferenceProvider` in `app/providers/` and implement
   `async def infer(self, request: InferenceRequest) -> dict`. Raise
   `ProviderError` on failure.
2. Register it in `app/providers/factory.py` under a new
   `INFERENCE_PROVIDER` value.
3. Add any provider-specific settings to `app/core/config.py` and
   `.env.example`.

Routes and handlers are unaffected — the factory is the only switch point.

## Project layout

```
app/
  main.py                 FastAPI app, routers, exception handlers
  core/                   config, logging, database, errors
  api/v1/                 health, projects, assets, capabilities, jobs, router
  models/                 SQLAlchemy: project, asset, job, enums, base
  schemas/                Pydantic v2 request/response models
  services/               projects, assets, storage, jobs (business logic)
  capabilities/           definitions, registry, context, handlers/
  providers/              base, mock, http, factory
  tools/                  runner, ffmpeg, imagemagick (subprocess wrappers)
  agent/                  catalog, schemas, planner (validation), llm, executor
  workers/                worker (arq), tasks, queue (enqueue helper)
alembic/                  async migration env + versions
tests/                    pytest suite (SQLite, eager jobs, mock provider)
frontend/index.html       placeholder page
```

## Future extensibility (out of scope for this MVP)

- Replace `LocalStorageService` with an S3/MinIO implementation (interface is
  ready; a commented `minio` service exists in `docker-compose.yml`).
- Lower-third caption overlays and brand-kit styling as new capabilities (the
  agent will offer them automatically once registered).
- Wire the remaining AI capabilities to real hosted endpoints.
- Authentication / multi-tenancy and a full editing UI.
