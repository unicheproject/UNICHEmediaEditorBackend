"""Storage abstraction.

LocalStorageService writes under STORAGE_DIR using the layout:
    projects/{project_id}/assets/{asset_id}/original/{safe_filename}
Reserved for later: .../derived/ and .../versions/.

The interface is intentionally narrow so an S3/MinIO implementation can be
dropped in without touching callers.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Protocol

from app.core.config import settings

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_CHUNK = 1024 * 1024


def safe_filename(name: str) -> str:
    """Sanitize an uploaded filename to a safe stored name."""
    base = Path(name).name  # strip any path components
    cleaned = _SAFE_NAME_RE.sub("_", base).strip("._") or "file"
    return cleaned[:255]


class StorageService(Protocol):
    def save_upload(
        self, project_id: uuid.UUID, asset_id: uuid.UUID, filename: str, src: BinaryIO
    ) -> tuple[str, int]: ...

    def get_path(self, storage_path: str) -> Path: ...

    def open_for_read(self, storage_path: str) -> BinaryIO: ...

    def delete_asset(self, project_id: uuid.UUID, asset_id: uuid.UUID) -> None: ...

    def calculate_checksum(self, storage_path: str) -> str: ...


class LocalStorageService:
    def __init__(self, root: str | None = None) -> None:
        self.root = Path(root or settings.storage_dir)

    def _asset_dir(self, project_id: uuid.UUID, asset_id: uuid.UUID) -> Path:
        return self.root / "projects" / str(project_id) / "assets" / str(asset_id)

    def save_upload(
        self, project_id: uuid.UUID, asset_id: uuid.UUID, filename: str, src: BinaryIO
    ) -> tuple[str, int]:
        """Persist an upload stream. Returns (relative_storage_path, size_bytes)."""
        dest_dir = self._asset_dir(project_id, asset_id) / "original"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename

        size = 0
        with dest.open("wb") as out:
            while chunk := src.read(_CHUNK):
                out.write(chunk)
                size += len(chunk)

        return str(dest.relative_to(self.root)), size

    def get_path(self, storage_path: str) -> Path:
        return self.root / storage_path

    def open_for_read(self, storage_path: str) -> BinaryIO:
        return self.get_path(storage_path).open("rb")

    def delete_asset(self, project_id: uuid.UUID, asset_id: uuid.UUID) -> None:
        asset_dir = self._asset_dir(project_id, asset_id)
        if asset_dir.exists():
            shutil.rmtree(asset_dir, ignore_errors=True)

    def calculate_checksum(self, storage_path: str) -> str:
        digest = hashlib.sha256()
        with self.open_for_read(storage_path) as fh:
            while chunk := fh.read(_CHUNK):
                digest.update(chunk)
        return digest.hexdigest()


def get_storage() -> StorageService:
    return LocalStorageService()
