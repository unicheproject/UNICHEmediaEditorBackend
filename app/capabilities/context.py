"""Execution context and result types shared by all capability handlers.

These generalize the handler boundary so a single interface serves both
provider-backed (JSON-returning) capabilities and local-tool capabilities that
produce output files.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.models.enums import MediaType


@dataclass
class JobContext:
    """Everything a handler needs to execute one job, without DB coupling."""

    capability_id: str
    params: dict[str, Any] = field(default_factory=dict)
    # Primary input asset (if any).
    input_path: str | None = None
    input_asset_meta: dict[str, Any] = field(default_factory=dict)
    # All resolved input paths (primary first, then extras for multi-input ops).
    input_paths: list[str] = field(default_factory=list)
    # Named secondary inputs resolved from any `*_asset_id` param
    # (e.g. subtitle_asset_id, music_asset_id, audio_asset_id) -> file path.
    named_input_paths: dict[str, str] = field(default_factory=dict)
    # Per-job scratch directory for tool output files.
    work_dir: Path | None = None
    project_id: uuid.UUID | None = None
    source_asset_id: uuid.UUID | None = None

    def out_path(self, filename: str) -> Path:
        """Absolute path inside the scratch dir for a named output file."""
        assert self.work_dir is not None, "work_dir is required for file-producing handlers"
        return self.work_dir / filename


@dataclass
class OutputFile:
    """A file produced by a handler, to be registered as a derived Asset."""

    path: Path
    filename: str
    media_type: MediaType


@dataclass
class HandlerResult:
    """Handler return value: JSON data for job.output + any produced files."""

    data: dict[str, Any] = field(default_factory=dict)
    outputs: list[OutputFile] = field(default_factory=list)
