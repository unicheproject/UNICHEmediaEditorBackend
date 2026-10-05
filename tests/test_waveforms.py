"""Bounded waveform generation, cache, ingest, lifecycle and authorization contracts."""

import asyncio
import io
import shutil
import subprocess
import uuid
import wave
from array import array
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select

from app.api.v1.assets import get_asset_probe_enqueuer
from app.core.database import async_session_factory
from app.main import app
from app.models.asset import Asset
from app.models.enums import MediaType, WaveformStatus
from app.models.job import Job
from app.scripts.backfill_asset_waveforms import backfill
from app.services import assets as assets_svc
from app.services import waveforms
from app.services.catalogue_client import get_catalogue_client
from app.services.storage import get_storage
from app.services.temp_assets import cleanup_expired_temp_assets
from app.tools import waveform as tool
from app.workers.tasks import run_asset_probe

API = "/api/v1"
requires_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")


def _wav_bytes() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(array("h", [8192] * 256 + [-16384] * 256 + [0] * 128).tobytes())
    return stream.getvalue()


async def _upload(client, pid, filename="tone.wav", data=None) -> dict:
    response = await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": (filename, _wav_bytes() if data is None else data)},
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _enrich(aid: str) -> None:
    await run_asset_probe({}, aid)


async def test_upload_queues_enrichment_and_read_is_pending(client, make_project) -> None:
    queued = AsyncMock()
    app.dependency_overrides[get_asset_probe_enqueuer] = lambda: queued
    pid = await make_project()
    asset = await _upload(client, pid)
    queued.assert_awaited_once_with(uuid.UUID(asset["id"]))
    assert asset["waveform_status"] == "pending"
    response = await client.get(f"{API}/assets/{asset['id']}/waveform")
    assert response.status_code == 202
    assert response.headers["retry-after"] == "5"
    assert response.json() == {"status": "pending", "reason_code": None}
    async with async_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Job)) == 0


@requires_ffmpeg
async def test_real_waveform_cache_and_rename(client, make_project, monkeypatch) -> None:
    asset = await _upload(client, await make_project())
    await _enrich(asset["id"])
    url = f"{API}/assets/{asset['id']}/waveform"
    response = await client.get(url)
    assert response.status_code == 200, response.text
    waveform = response.json()
    assert waveform == {
        "status": "available", "version": 1, "channels": 1, "sample_rate": 8000,
        "bucket_samples": 256, "bucket_count": 3, "duration_seconds": 0.08,
        "peaks": [[0.25, 0.25], [-0.5, -0.5], [0.0, 0.0]],
    }
    assert "path" not in response.text and "checksum" not in response.text
    decoder = AsyncMock(side_effect=AssertionError("Cached reads must never decode"))
    monkeypatch.setattr(waveforms, "generate_waveform", decoder)
    assert (await client.get(url)).json() == waveform
    rename = await client.patch(
        f"{API}/assets/{asset['id']}", json={"original_filename": "Approved narration"},
    )
    assert rename.status_code == 200
    await _enrich(asset["id"])
    assert (await client.get(url)).json() == waveform
    decoder.assert_not_awaited()
    async with async_session_factory() as session:
        row = await session.get(Asset, uuid.UUID(asset["id"]))
        assert row is not None and waveforms.artifact_path(row).is_file()
        assert await session.scalar(select(func.count()).select_from(Asset)) == 1


def test_stream_reduction_bound_chunk_invariance_and_normalization() -> None:
    source = array("h", [-32768, 32767, -100, 100] * 150000)
    results = []
    for size in (256, 32768):
        accumulator = tool.EnvelopeAccumulator()
        for offset in range(0, len(source), size):
            accumulator.add(source[offset:offset + size])
            assert len(accumulator.peaks) <= 1024
        results.append(accumulator.finish())
    assert results[0] == results[1]
    result = results[0]
    assert result.bucket_count <= 1024
    assert result.duration_seconds == len(source) / 8000
    assert all(-1 <= lo <= hi <= 1 for lo, hi in result.peaks)
    assert len(result.model_dump_json().encode()) < waveforms.MAX_ARTIFACT_BYTES


@pytest.mark.parametrize("samples", [1, 256, 1024 * 256, 1024 * 256 + 1])
def test_partial_bucket_boundaries(samples: int) -> None:
    accumulator = tool.EnvelopeAccumulator()
    accumulator.add(array("h", [0]) * samples)
    result = accumulator.finish()
    assert result.bucket_count <= 1024
    assert result.peaks == [(0, 0)] * result.bucket_count


@requires_ffmpeg
async def test_corrupt_media_is_unavailable_but_downloadable(client, make_project) -> None:
    asset = await _upload(client, await make_project(), data=b"not audio")
    await _enrich(asset["id"])
    response = await client.get(f"{API}/assets/{asset['id']}/waveform")
    assert response.status_code == 200
    assert response.json() == {"status": "unavailable", "reason_code": "generation_failed"}
    assert (await client.get(f"{API}/assets/{asset['id']}/download")).content == b"not audio"


async def test_unsupported_media_is_not_a_fake_waveform(client, make_project) -> None:
    asset = await _upload(client, await make_project(), "image.png", b"image bytes")
    assert asset["waveform_status"] == "unavailable"
    response = await client.get(f"{API}/assets/{asset['id']}/waveform")
    assert response.json() == {"status": "unavailable", "reason_code": "unsupported_media"}


@requires_ffmpeg
@pytest.mark.parametrize("with_audio", [False, True])
async def test_video_first_audio_stream_or_silent_unavailable(
    client, make_project, tmp_path, with_audio,
) -> None:
    source = tmp_path / "video.mp4"
    args = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
            "color=red:size=32x24:rate=10:duration=0.2"]
    if with_audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=0.2", "-c:a", "aac"]
    subprocess.run([*args, "-c:v", "mpeg4", str(source)], check=True, capture_output=True)
    asset = await _upload(client, await make_project(), source.name, source.read_bytes())
    await _enrich(asset["id"])
    result = (await client.get(f"{API}/assets/{asset['id']}/waveform")).json()
    if with_audio:
        assert result["status"] == "available" and result["bucket_count"] > 0
        assert all(-1 <= lo <= hi <= 1 for lo, hi in result["peaks"])
    else:
        assert result == {"status": "unavailable", "reason_code": "no_audio"}


@requires_ffmpeg
async def test_auth_delete_and_missing_binary(client, make_project, monkeypatch) -> None:
    asset = await _upload(client, await make_project())
    await _enrich(asset["id"])
    url = f"{API}/assets/{asset['id']}/waveform"
    catalogue = app.dependency_overrides[get_catalogue_client]()
    original = catalogue.get_project
    monkeypatch.setattr(catalogue, "get_project", AsyncMock(return_value=None))
    denied = await client.get(url)
    assert denied.status_code == 404 and "peaks" not in denied.text
    monkeypatch.setattr(catalogue, "get_project", original)
    async with async_session_factory() as session:
        row = await session.get(Asset, uuid.UUID(asset["id"]))
        assert row is not None
        get_storage().get_path(row.storage_path).unlink()
    assert (await client.get(url)).status_code == 404
    assert (await client.delete(f"{API}/assets/{asset['id']}")).status_code == 204
    assert (await client.get(url)).status_code == 404
    assert (await client.get(f"{API}/assets/not-a-uuid/waveform")).status_code == 422


@requires_ffmpeg
async def test_temp_cleanup_removes_sidecar_and_expiry_cannot_be_bypassed(
    client, make_project,
) -> None:
    asset = await _upload(client, await make_project())
    await _enrich(asset["id"])
    async with async_session_factory() as session:
        row = await session.get(Asset, uuid.UUID(asset["id"]))
        assert row is not None
        path = waveforms.artifact_path(row)
        assert path.is_file()
        row.is_intermediate = True
        row.temp_expires_at = datetime.now(UTC) - timedelta(hours=73)
        await session.commit()
        assert await cleanup_expired_temp_assets(session) == 1
        assert not path.exists()
        assert row.waveform_status == "available"  # retained audit metadata
    response = await client.get(f"{API}/assets/{asset['id']}/waveform")
    assert response.status_code == 410


@requires_ffmpeg
async def test_derived_enrichment_is_nonfatal_and_preserves_asset_output(
    client, make_project, tmp_path, monkeypatch,
) -> None:
    pid = uuid.UUID(await make_project())
    source = tmp_path / "derived.wav"
    source.write_bytes(_wav_bytes())
    async with async_session_factory() as session:
        output = await assets_svc.create_derived_asset(
            session, project_id=pid, source_asset_id=None, src_path=source,
            filename=source.name, media_type=MediaType.audio,
        )
        assert output.waveform_status == "available"
        monkeypatch.setattr(waveforms, "generate_waveform", AsyncMock(side_effect=RuntimeError()))
        failed_preview = await assets_svc.create_derived_asset(
            session, project_id=pid, source_asset_id=output.id, src_path=source,
            filename="still-successful.wav", media_type=MediaType.audio,
        )
        assert failed_preview.waveform_status == "unavailable"
        assert failed_preview.technical_metadata_status == "available"
        assert get_storage().get_path(failed_preview.storage_path).is_file()


@requires_ffmpeg
async def test_missing_artifact_backfill_repairs_and_is_project_scoped(
    client, make_project, monkeypatch,
) -> None:
    pid = await make_project()
    asset = await _upload(client, pid)
    other = await _upload(client, await make_project())
    assert await backfill(project_id=uuid.UUID(pid), batch_size=1) == {
        "available": 1, "unavailable": 0, "skipped": 0,
    }
    async with async_session_factory() as session:
        row = await session.get(Asset, uuid.UUID(asset["id"]))
        assert row is not None
        waveforms.artifact_path(row).unlink()
    missing = await client.get(f"{API}/assets/{asset['id']}/waveform")
    assert missing.json() == {"status": "unavailable", "reason_code": "artifact_missing"}
    assert (await client.get(f"{API}/assets/{other['id']}")).json()["waveform_status"] == "pending"
    assert (await backfill(project_id=uuid.UUID(pid), batch_size=1))["available"] == 1
    decoder = AsyncMock(side_effect=AssertionError("Backfill must reuse valid cache"))
    monkeypatch.setattr(waveforms, "generate_waveform", decoder)
    assert (await backfill(project_id=uuid.UUID(pid), batch_size=1))["available"] == 1
    decoder.assert_not_awaited()
    with pytest.raises(ValueError):
        await backfill(project_id=None, batch_size=0)


@requires_ffmpeg
async def test_checksum_change_does_not_reuse_old_waveform(client, make_project) -> None:
    asset = await _upload(client, await make_project())
    await _enrich(asset["id"])
    async with async_session_factory() as session:
        row = await session.get(Asset, uuid.UUID(asset["id"]))
        assert row is not None
        row.checksum_sha256 = "0" * 64
        await session.commit()
    result = (await client.get(f"{API}/assets/{asset['id']}/waveform")).json()
    assert result == {"status": "unavailable", "reason_code": "identity_mismatch"}


@requires_ffmpeg
async def test_storage_failure_is_nonfatal(client, make_project, monkeypatch) -> None:
    asset = await _upload(client, await make_project())
    monkeypatch.setattr(waveforms, "_write_artifact", lambda *_: (_ for _ in ()).throw(OSError()))
    await _enrich(asset["id"])
    row = (await client.get(f"{API}/assets/{asset['id']}")).json()
    assert row["waveform_status"] == "unavailable"
    assert row["technical_metadata_status"] == "available"


@requires_ffmpeg
async def test_waveform_failure_cannot_fail_successful_media_job(
    client, make_project, monkeypatch,
) -> None:
    asset = await _upload(client, await make_project())
    monkeypatch.setattr(
        waveforms, "generate_waveform",
        AsyncMock(side_effect=tool.WaveformGenerationError("Decoder timeout")),
    )
    response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": "audio.trim", "asset_id": asset["id"],
            "input": {"start": 0, "end": 0.064},
        },
    )
    assert response.status_code == 201, response.text
    job = (await client.get(f"{API}/jobs/{response.json()['id']}")).json()
    assert job["status"] == "succeeded", job.get("error")
    outputs = job["output"]["outputs"]
    assert len(outputs) == 1
    output = (await client.get(f"{API}/assets/{outputs[0]['asset_id']}")).json()
    assert output["waveform_status"] == "unavailable"
    assert output["waveform_reason"] == "generation_failed"
    assert output["technical_metadata_status"] == "available"
    assert (await client.get(f"{API}/assets/{output['id']}/download")).status_code == 200


@requires_ffmpeg
@pytest.mark.parametrize("payload", [b"invalid JSON", b"x" * (64 * 1024 + 1)])
async def test_corrupt_or_oversized_artifact_is_not_returned(
    client, make_project, payload,
) -> None:
    asset = await _upload(client, await make_project())
    await _enrich(asset["id"])
    async with async_session_factory() as session:
        row = await session.get(Asset, uuid.UUID(asset["id"]))
        assert row is not None
        waveforms.artifact_path(row).write_bytes(payload)
    response = await client.get(f"{API}/assets/{asset['id']}/waveform")
    assert response.status_code == 200
    assert response.json() == {"status": "unavailable", "reason_code": "artifact_missing"}


@pytest.mark.parametrize("cancel", [False, True])
async def test_decoder_timeout_and_cancellation_kill_child(monkeypatch, tmp_path, cancel) -> None:
    entered = asyncio.Event()

    class Process:
        returncode = None
        killed = False

        @property
        def stdout(self):
            return self

        async def read(self, size):
            assert size == tool.PCM_CHUNK_BYTES
            entered.set()
            await asyncio.Event().wait()

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            return self.returncode

    proc = Process()
    spawn = AsyncMock(return_value=proc)
    monkeypatch.setattr(tool.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(tool, "WAVEFORM_TIMEOUT_SECONDS", 0.01)
    task = asyncio.create_task(tool.generate_waveform(tmp_path / "file.wav"))
    await entered.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else tool.WaveformGenerationError):
        await task
    assert proc.killed
    args = spawn.call_args.args
    assert args[args.index("-map") + 1] == "0:a:0"
    assert args[args.index("-ac") + 1] == "1"
    assert spawn.call_args.kwargs["stderr"] == asyncio.subprocess.DEVNULL


def test_waveform_openapi_models_are_typed_and_bounded() -> None:
    from app.schemas.waveform import WaveformAvailable

    schema = WaveformAvailable.model_json_schema()
    assert schema["properties"]["peaks"]["maxItems"] == 1024
    route = app.openapi()["paths"][f"{API}/assets/{{asset_id}}/waveform"]["get"]
    assert "200" in route["responses"] and "202" in route["responses"]
    assert WaveformStatus.available.value == "available"
