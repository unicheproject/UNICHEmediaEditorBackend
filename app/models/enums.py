"""Enumerations shared across models and schemas."""

from __future__ import annotations

import enum


class MediaType(enum.StrEnum):
    image = "image"
    audio = "audio"
    video = "video"
    subtitle = "subtitle"
    unknown = "unknown"


class TechnicalMetadataStatus(enum.StrEnum):
    pending = "pending"
    available = "available"
    unavailable = "unavailable"


class WaveformStatus(enum.StrEnum):
    pending = "pending"
    available = "available"
    unavailable = "unavailable"


class WaveformReason(enum.StrEnum):
    unsupported_media = "unsupported_media"
    no_audio = "no_audio"
    generation_failed = "generation_failed"
    artifact_missing = "artifact_missing"
    identity_mismatch = "identity_mismatch"


class JobStatus(enum.StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class CostClass(enum.StrEnum):
    local_cpu = "local_cpu"
    hosted_ai = "hosted_ai"
    future_gpu = "future_gpu"
    deterministic = "deterministic"


class AgentPlanStatus(enum.StrEnum):
    proposed = "proposed"
    approved = "approved"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
