"""File helpers for safe paths, sessions, and JSON persistence."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any


def make_session_id(prefix: str = "session") -> str:
    """Create a short id that is easy to read in the UI."""

    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def slugify(value: str) -> str:
    """Turn a file name into a safe lower-case folder name."""

    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "item"


def ensure_parent(path: Path) -> None:
    """Create the parent folder for a file path when it does not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON to disk using a readable indent level."""

    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON from disk and return a fallback value when the file is missing."""

    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
