"""JSON helpers used by the output writer and sample data scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .file_utils import read_json, write_json


def load_json(path: str | Path, default: Any = None) -> Any:
    """Load a JSON file from a string path or a Path object."""

    return read_json(Path(path), default=default)


def save_json(path: str | Path, payload: Any) -> None:
    """Save a JSON payload to disk."""

    write_json(Path(path), payload)
