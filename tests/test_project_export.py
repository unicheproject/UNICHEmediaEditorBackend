from __future__ import annotations

import io
import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.api.v1.jobs import get_enqueuer
from app.capabilities import registry
from app.capabilities.validation import (
    CapabilityInputValidationError,
    validate_capability_input,
)
from app.composition.export_presets import (
    ADVANCED_EXPORT_OPTIONS,
    EXPORT_PRESET_VERSION,
    EXPORT_PRESETS,
    resolve_effective_settings,
)
from app.core.database import async_session_factory
from app.main import app
from app.models.asset import Asset
from app.models.composition import ProjectCompositionRevision
from app.models.job import Job
from app.services.jobs import execute_job
from app.services.storage import get_storage
from app.tools import ffmpeg
from app.tools.compose_policy import LOSSLESS_COMPOSE_RENDER_POLICY

API = "/api/v1"
HAVE_MEDIA_TOOLS = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
NEED_MEDIA_TOOLS = pytest.mark.skipif(
    not HAVE_MEDIA_TOOLS, reason="project export integration needs ffmpeg and ffprobe"
)
LOSSY_ENCODERS = {
    "libx264",
    "libx265",
    "libvpx-vp9",
    "aac",
    "libmp3lame",
    "libopus",
    "libvorbis",
}


def _make_video(path: Path, width: int = 96, height: int = 64) -> None:
    process = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=navy:s={width}x{height}:r=10:d=0.4",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=0.4",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def _probe(path: Path) -> dict:
    process = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=format_name,duration:stream=codec_type,codec_name,width,height,pix_fmt,channels",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


async def _upload_video(
    client: AsyncClient, project_id: str, path: Path, name: str = "clip.mp4"
) -> str:
    response = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": (name, io.BytesIO(path.read_bytes()), "video/mp4")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _document(
    asset_ids: list[str], *, width: int = 96, height: int = 64
) -> dict:
    return {
        "schema_version": 1,
        "canvas": {"width": width, "height": height},
        "tracks": [
            {
                "id": str(uuid.uuid4()),
                "kind": "video",
                "clips": [
                    {"id": str(uuid.uuid4()), "asset_id": asset_id}
                    for asset_id in asset_ids
                ],
            }
        ],
        "audio_bed": None,
    }


async def _save(
    client: AsyncClient, project_id: str, document: dict, *, etag: str | None = None
) -> dict:
    headers = {"If-Match": etag} if etag else {"If-None-Match": "*"}
    response = await client.put(
        f"{API}/projects/{project_id}/composition",
        json=document,
        headers=headers,
    )
    assert response.status_code in {200, 201}, response.text
    return response.json()


async def _leave_queued(_job_id: uuid.UUID) -> None:
    return None


async def _queue_export(client: AsyncClient, project_id: str, revision_id: str, **values):
    app.dependency_overrides[get_enqueuer] = lambda: _leave_queued
    response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": "project.export",
            "project_id": project_id,
            "input": {"composition_revision_id": revision_id, **values},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_export_preset_registry_and_effective_settings_are_canonical() -> None:
    assert {preset.value for preset in EXPORT_PRESETS} == {
        "web", "social", "preview", "archival", "advanced"
    }
    assert EXPORT_PRESET_VERSION == 1

    web = resolve_effective_settings(
        {"preset": "web"}, canvas_width=640, canvas_height=480
    )
    assert (web.container.value, web.video_codec.value, web.audio_codec.value) == (
        "mp4", "libx264", "aac"
    )
    assert (web.profile, web.pixel_format.value, web.crf) == (
        "high", "yuv420p", 22
    )
    assert (web.max_width, web.max_height, web.fit.value, web.faststart) == (
        1920, 1080, "fit", True
    )
    assert (web.audio_bitrate_kbps, web.audio_channels) == (128, 2)

    square = resolve_effective_settings(
        {"preset": "social", "social_format": "square"},
        canvas_width=640,
        canvas_height=480,
    )
    portrait = resolve_effective_settings(
        {"preset": "social", "social_format": "portrait"},
        canvas_width=640,
        canvas_height=480,
    )
    assert (square.target_width, square.target_height) == (1080, 1080)
    assert (portrait.target_width, portrait.target_height) == (1080, 1920)
    assert square.fit.value == portrait.fit.value == "pad"
    assert square.crf == portrait.crf == 21
    assert square.audio_bitrate_kbps == portrait.audio_bitrate_kbps == 128

    preview = resolve_effective_settings(
        {"preset": "preview"}, canvas_width=640, canvas_height=480
    )
    assert (preview.container.value, preview.max_width, preview.max_height) == (
        "mp4", 1280, 720
    )
    assert (preview.crf, preview.audio_bitrate_kbps, preview.faststart) == (
        28, 96, True
    )

    archival = resolve_effective_settings(
        {"preset": "archival"}, canvas_width=640, canvas_height=480
    )
    assert archival.composition_policy.value == "lossless"
    assert archival.container.value == "mkv"
    assert archival.video_codec.value == "ffv1"
    assert archival.audio_codec.value == "flac"
    assert archival.pixel_format.value == LOSSLESS_COMPOSE_RENDER_POLICY.pixel_format.value
    assert archival.crf is None and archival.audio_bitrate_kbps is None
    assert archival.faststart is False

    assert ADVANCED_EXPORT_OPTIONS == {
        "mp4": {"video_codec": ["libx264"], "audio_codec": ["aac"]},
        "mkv": {"video_codec": ["libx264"], "audio_codec": ["aac"]},
    }
    advanced = resolve_effective_settings(
        {
            "preset": "advanced", "format": "mkv",
            "video_codec": "libx264", "audio_codec": "aac",
            "max_width": 640, "max_height": 480, "crf": 24,
            "audio_bitrate_kbps": 160,
        },
        canvas_width=1920,
        canvas_height=1080,
    )
    assert advanced.container.value == "mkv"
    assert advanced.crf == 24 and advanced.audio_bitrate_kbps == 160


@pytest.mark.parametrize(
    "value",
    [
        {"composition_revision_id": str(uuid.uuid4()), "preset": "bad"},
        {"composition_revision_id": str(uuid.uuid4()), "preset": "social"},
        {
            "composition_revision_id": str(uuid.uuid4()),
            "preset": "web",
            "social_format": "portrait",
        },
        {
            "composition_revision_id": str(uuid.uuid4()),
            "preset": "archival",
            "format": "mp4",
        },
        {
            "composition_revision_id": str(uuid.uuid4()),
            "preset": "advanced",
            "format": "avi",
            "video_codec": "libx264",
            "audio_codec": "aac",
        },
        {
            "composition_revision_id": str(uuid.uuid4()),
            "preset": "advanced",
            "format": "mp4",
            "video_codec": "libx265",
            "audio_codec": "aac",
        },
        {
            "composition_revision_id": str(uuid.uuid4()),
            "preset": "advanced",
            "format": "webm",
            "video_codec": "libx264",
            "audio_codec": "aac",
        },
        {
            "composition_revision_id": str(uuid.uuid4()),
            "preset": "advanced",
            "format": "mp4",
            "video_codec": "libx264",
            "audio_codec": "aac",
            "ffmpeg_args": "-c:v copy",
        },
        {
            "composition_revision_id": str(uuid.uuid4()),
            "preset": "advanced",
            "format": "mp4",
            "video_codec": "libx264",
            "audio_codec": "aac",
            "crf": 60,
        },
        {
            "composition_revision_id": str(uuid.uuid4()),
            "preset": "advanced",
            "format": "mp4",
            "video_codec": "libx264",
            "audio_codec": "aac",
            "max_width": 641,
        },
    ],
)
def test_project_export_canonical_schema_rejects_invalid_selections(value: dict) -> None:
    with pytest.raises(CapabilityInputValidationError):
        validate_capability_input(registry.get("project.export"), value)


@NEED_MEDIA_TOOLS
async def test_project_export_schema_and_source_less_async_job_snapshot(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    capability = (await client.get(f"{API}/capabilities/project.export")).json()
    schema = capability["input_schema"]
    assert capability["supported_media_types"] == []
    assert schema["properties"]["preset"]["enum"] == [
        "web", "social", "preview", "archival", "advanced"
    ]
    assert schema["properties"]["social_format"]["x-option-constraints"][
        "allowed_values"
    ] == {"social": ["square", "portrait"]}
    assert schema["properties"]["video_codec"]["enum"] == ["libx264"]
    assert "ffmpeg_args" not in schema["properties"]
    assert "execution_snapshot" not in schema["properties"]

    project_id = await make_project("Export")
    video_path = tmp_path / "source.mp4"
    _make_video(video_path)
    asset_id = await _upload_video(client, project_id, video_path)
    revision = await _save(client, project_id, _document([asset_id, asset_id, asset_id]))
    job = await _queue_export(
        client, project_id, revision["revision_id"], preset="web"
    )
    assert job["status"] == "queued"
    assert job["asset_id"] is None
    assert job["project_id"] == project_id
    async with async_session_factory() as session:
        persisted_job = await session.get(Job, uuid.UUID(job["id"]))
        assert persisted_job is not None
        assert persisted_job.project_id == uuid.UUID(project_id)
    snapshot = job["input"]["execution_snapshot"]
    assert snapshot["composition_revision_id"] == revision["revision_id"]
    assert snapshot["composition_document_hash"] == revision["document_hash"]
    assert snapshot["composition_compiler_version"] == 1
    assert snapshot["preset"] == "web"
    assert snapshot["preset_version"] == EXPORT_PRESET_VERSION
    assert snapshot["effective_settings"]["container"] == "mp4"
    assert [item["asset_id"] for item in snapshot["ordered_video_sources"]] == [
        asset_id, asset_id, asset_id
    ]
    assert len({item["checksum_sha256"] for item in snapshot["ordered_video_sources"]}) == 1


@NEED_MEDIA_TOOLS
async def test_export_binds_queued_job_to_exact_immutable_revision(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    project_id = await make_project("Revision pinned export")
    video_path = tmp_path / "source.mp4"
    _make_video(video_path)
    asset_id = await _upload_video(client, project_id, video_path)
    revision_n = await _save(client, project_id, _document([asset_id], width=96, height=64))
    job = await _queue_export(
        client, project_id, revision_n["revision_id"], preset="web"
    )

    current = await client.get(f"{API}/projects/{project_id}/composition")
    revision_n1 = await _save(
        client,
        project_id,
        _document([asset_id], width=64, height=32),
        etag=current.headers["etag"],
    )
    assert revision_n1["revision_id"] != revision_n["revision_id"]

    async with async_session_factory() as session:
        executed = await execute_job(session, uuid.UUID(job["id"]))
        assert executed.status.value == "succeeded", executed.error
        output_id = uuid.UUID(executed.output["outputs"][0]["asset_id"])
        asset = await session.get(Asset, output_id)
        assert asset is not None
        assert (asset.width, asset.height) == (96, 64)
        assert asset.source_asset_id == uuid.UUID(asset_id)
        assert asset.is_intermediate is False
        assert asset.temp_expires_at is None
        assert asset.technical_metadata_status == "available"
        assert executed.output["composition_revision_id"] == revision_n["revision_id"]
        assert executed.output["video_encode_passes"] == 2
        revision_row = await session.get(
            ProjectCompositionRevision, uuid.UUID(revision_n["revision_id"])
        )
        assert revision_row is not None
        assert revision_row.document["canvas"] == {"width": 96, "height": 64}

    outputs = executed.output["outputs"]
    assert len(outputs) == 1
    normal_assets = await client.get(f"{API}/projects/{project_id}/assets")
    assert outputs[0]["asset_id"] in {asset["id"] for asset in normal_assets.json()}
    downloaded = await client.get(f"{API}/assets/{outputs[0]['asset_id']}/download")
    assert downloaded.status_code == 200


@NEED_MEDIA_TOOLS
async def test_archival_export_uses_lossless_policy_from_first_compose_invocation(
    client: AsyncClient, make_project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id = await make_project("Archival")
    video_path = tmp_path / "source.mp4"
    _make_video(video_path)
    asset_id = await _upload_video(client, project_id, video_path)
    revision = await _save(client, project_id, _document([asset_id, asset_id]))
    job = await _queue_export(
        client, project_id, revision["revision_id"], preset="archival"
    )
    commands: list[list[str]] = []
    real_run = ffmpeg.runner.run

    async def capture(args: list[str], *, timeout: float = 600.0) -> str:
        commands.append(list(args))
        return await real_run(args, timeout=timeout)

    monkeypatch.setattr(ffmpeg.runner, "run", capture)
    async with async_session_factory() as session:
        executed = await execute_job(session, uuid.UUID(job["id"]))
        assert executed.status.value == "succeeded", executed.error
        descriptor = executed.output["outputs"][0]
        output_asset = await session.get(Asset, uuid.UUID(descriptor["asset_id"]))
        assert output_asset is not None
        output_path = get_storage().get_path(output_asset.storage_path)
        probed = _probe(output_path)
        streams = {stream["codec_type"]: stream for stream in probed["streams"]}
        assert "matroska" in probed["format"]["format_name"]
        assert streams["video"]["codec_name"] == "ffv1"
        assert streams["audio"]["codec_name"] == "flac"
        assert streams["video"]["pix_fmt"] == "yuv444p16le"
        assert descriptor["filename"].endswith(".mkv")
        assert output_asset.is_intermediate is False
        assert output_asset.temp_expires_at is None
        assert output_asset.technical_metadata_status == "available"

    command_count = len(commands)
    original_output = executed.output
    original_snapshot = executed.input
    async with async_session_factory() as session:
        duplicate = await execute_job(session, uuid.UUID(job["id"]))
        assert duplicate.status.value == "succeeded"
        assert duplicate.output == original_output
        assert duplicate.input == original_snapshot
        published = list(await session.scalars(select(Asset).where(
            Asset.project_id == uuid.UUID(project_id), Asset.source_asset_id.is_not(None),
        )))
        assert len(published) == 1
        assert published[0].id == uuid.UUID(descriptor["asset_id"])
    assert len(commands) == command_count  # no second compose, probe or waveform decode

    encoders: set[str] = set()
    for command in commands:
        for index, value in enumerate(command[:-1]):
            if value in {"-c", "-codec", "-c:v", "-c:a", "-codec:v", "-codec:a"}:
                encoders.add(command[index + 1])
    assert "ffv1" in encoders and "flac" in encoders
    assert "copy" in encoders
    assert not encoders.intersection(LOSSY_ENCODERS)
    assert executed.output["video_encode_passes"] == 2
    assert len(executed.output["outputs"]) == 1


@pytest.mark.parametrize(
    ("preset", "extra", "expected_flags"),
    [
        ("web", {}, ["-crf", "22", "-b:a", "128k", "-ac", "2", "-movflags", "+faststart"]),
        ("preview", {}, ["-crf", "28", "-b:a", "96k", "-movflags", "+faststart"]),
        (
            "social",
            {"social_format": "square"},
            ["pad=1080:1080", "-crf", "21", "-b:a", "128k", "+faststart"],
        ),
    ],
)
async def test_delivery_transform_uses_profile_scaling_and_settings(
    preset: str,
    extra: dict,
    expected_flags: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = resolve_effective_settings(
        {"preset": preset, **extra}, canvas_width=640, canvas_height=480
    )
    commands: list[list[str]] = []

    async def fake_run(args: list[str], *, timeout: float = 600.0) -> str:
        commands.append(args)
        return ""

    monkeypatch.setattr(ffmpeg.runner, "run", fake_run)
    await ffmpeg.export_delivery_transform("source.mp4", "out.mp4", settings)
    command = commands[0]
    joined = " ".join(command)
    assert "libx264" in command
    assert "-pix_fmt" in command and command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert all(flag in joined for flag in expected_flags)
    assert "force_original_aspect_ratio=decrease" in joined
    if preset == "web":
        assert "min(iw,1920)" in joined and "min(ih,1080)" in joined
    if preset == "preview":
        assert "min(iw,1280)" in joined and "min(ih,720)" in joined


@NEED_MEDIA_TOOLS
async def test_export_submission_rejects_missing_and_cross_project_revisions(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    project_id = await make_project("Export A")
    other_project_id = await make_project("Export B")
    absent = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": "project.export",
            "project_id": project_id,
            "input": {"composition_revision_id": str(uuid.uuid4()), "preset": "web"},
        },
    )
    assert absent.status_code == 404

    # A revision owned by another project cannot be bound to this Job.
    source_path = tmp_path / "foreign.mp4"
    _make_video(source_path)
    video_id = await _upload_video(client, other_project_id, source_path, "foreign.mp4")
    revision = await _save(client, other_project_id, _document([video_id]))
    response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": "project.export",
            "project_id": project_id,
            "input": {
                "composition_revision_id": revision["revision_id"],
                "preset": "web",
            },
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "composition_revision_not_found"


async def test_project_export_is_not_in_agent_retry_reuse_allowlist() -> None:
    from app.agent.reuse import REUSE_VERSIONS

    assert "project.export" not in REUSE_VERSIONS
