"""Enumerations shared across models and schemas."""

from __future__ import annotations

import enum


class MediaType(enum.StrEnum):
    image = "image"
    audio = "audio"
    video = "video"
    unknown = "unknown"


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
