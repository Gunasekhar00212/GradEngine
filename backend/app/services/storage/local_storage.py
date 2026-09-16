"""Local file storage backed by the repository data directory."""

from __future__ import annotations

import shutil
from pathlib import Path

from backend.app.core.config import PATHS
from backend.app.services.storage.base import StorageBackend


class LocalStorage(StorageBackend):
    """Persist files on disk using relative storage keys."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir or PATHS.data_dir)

    def get_file_path(self, key: str) -> Path:
        return self.root_dir / key

    def save_file(self, source: bytes | str | Path, key: str) -> str:
        path = self.get_file_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(source, bytes):
            path.write_bytes(source)
        else:
            shutil.copy2(Path(source), path)

        return str(path)

    def get_file(self, key: str) -> bytes:
        return self.get_file_path(key).read_bytes()

    def delete_file(self, key: str) -> None:
        path = self.get_file_path(key)
        if path.exists():
            path.unlink()

    def file_exists(self, key: str) -> bool:
        return self.get_file_path(key).exists()
