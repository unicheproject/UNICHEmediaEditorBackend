"""Non-fatal, cached waveform enrichment using Asset-local preview sidecars."""

import asyncio
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.asset import Asset
from app.models.enums import MediaType, WaveformReason, WaveformStatus
from app.schemas.waveform import WaveformAvailable, WaveformResponse, WaveformState
from app.services.storage import get_storage
from app.tools.waveform import generate_waveform

logger = get_logger(__name__)
MAX_ARTIFACT_BYTES = 64 * 1024
ARTIFACT_FILENAME = "waveform.v1.json"


class WaveformArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checksum_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    waveform: WaveformAvailable


def artifact_path(asset: Asset) -> Path:
    return get_storage().get_path(asset.storage_path).parent.parent / "waveform" / ARTIFACT_FILENAME


def _read_artifact(asset: Asset) -> tuple[WaveformAvailable | None, WaveformReason]:
    try:
        with artifact_path(asset).open("rb") as stream:
            payload = stream.read(MAX_ARTIFACT_BYTES + 1)
        if len(payload) > MAX_ARTIFACT_BYTES:
            return None, WaveformReason.artifact_missing
        artifact = WaveformArtifact.model_validate_json(payload)
        if artifact.checksum_sha256 != asset.checksum_sha256:
            return None, WaveformReason.identity_mismatch
        return artifact.waveform, WaveformReason.artifact_missing
    except (OSError, ValueError):
        return None, WaveformReason.artifact_missing


def _write_artifact(asset: Asset, waveform: WaveformAvailable) -> None:
    payload = WaveformArtifact(
        checksum_sha256=asset.checksum_sha256, waveform=waveform,
    ).model_dump_json().encode("utf-8")
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ValueError("Waveform artifact exceeds its bound")
    path = artifact_path(asset)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid.uuid4()}.json")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def initial_waveform_state(media_type: MediaType) -> tuple[WaveformStatus, WaveformReason | None]:
    if media_type in (MediaType.audio, MediaType.video):
        return WaveformStatus.pending, None
    return WaveformStatus.unavailable, WaveformReason.unsupported_media


def read_waveform(asset: Asset) -> WaveformResponse:
    """Read only: never decodes, schedules work or changes lifecycle state."""
    from app.services.assets import ensure_asset_binary_available

    ensure_asset_binary_available(asset)
    if asset.media_type not in (MediaType.audio, MediaType.video):
        return WaveformState(status="unavailable", reason_code=WaveformReason.unsupported_media)
    if asset.waveform_status == WaveformStatus.pending:
        return WaveformState(status="pending")
    if asset.waveform_status == WaveformStatus.available:
        waveform, reason = _read_artifact(asset)
        return waveform or WaveformState(status="unavailable", reason_code=reason)
    return WaveformState(
        status="unavailable",
        reason_code=WaveformReason(asset.waveform_reason or WaveformReason.generation_failed),
    )


async def enrich_asset_waveform(
    session: AsyncSession, asset_id: uuid.UUID,
) -> WaveformStatus | None:
    """Generate once under a row lock; preview/storage failures cannot fail media Jobs.

    Cleanup also locks Asset rows with SKIP LOCKED, preventing it from removing
    media midway through generation or leaving a newly written orphan sidecar.
    """
    try:
        return await _enrich(session, asset_id)
    except Exception as exc:  # noqa: BLE001 - enrichment is strictly best-effort
        await session.rollback()
        logger.warning("Waveform enrichment failed for asset %s (%s)", asset_id, type(exc).__name__)
        return None


async def _enrich(session: AsyncSession, asset_id: uuid.UUID) -> WaveformStatus | None:
    asset = await session.scalar(
        select(Asset).where(Asset.id == asset_id).with_for_update()
        .execution_options(populate_existing=True)
    )
    if asset is None or asset.deleted_at is not None or asset.physical_expired_at is not None:
        return None
    state, reason = initial_waveform_state(asset.media_type)
    if state == WaveformStatus.pending and asset.has_audio_stream is False:
        state, reason = WaveformStatus.unavailable, WaveformReason.no_audio
    if state == WaveformStatus.pending:
        cached, _ = _read_artifact(asset)
        try:
            source = get_storage().get_path(asset.storage_path)
            if not source.is_file():
                raise ValueError("Original media missing")
            if cached is None:
                checksum = await asyncio.to_thread(
                    get_storage().calculate_checksum, asset.storage_path,
                )
                if checksum != asset.checksum_sha256:
                    raise ValueError("Source identity mismatch")
                waveform = await generate_waveform(source)
                await asyncio.to_thread(_write_artifact, asset, waveform)
            state, reason = WaveformStatus.available, None
        except Exception as exc:  # noqa: BLE001 - record only a safe failure category
            logger.warning(
                "Waveform unavailable for asset %s (%s)", asset_id, type(exc).__name__,
            )
            state, reason = WaveformStatus.unavailable, WaveformReason.generation_failed
    asset.waveform_status = state.value
    asset.waveform_reason = reason.value if reason else None
    await session.commit()
    return state
