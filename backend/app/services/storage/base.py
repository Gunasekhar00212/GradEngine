"""Storage interface for local files and future object storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Common storage operations used by ingestion and processing services."""

    @abstractmethod
    def save_file(self, source: bytes | str | Path, key: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_file(self, key: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def delete_file(self, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def file_exists(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_file_path(self, key: str) -> Path:
        raise NotImplementedError
