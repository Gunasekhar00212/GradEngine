"""Convert extracted answer data into the structured Answer JSON format."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.utils.file_utils import write_json


class AnswerJsonService:
    """Build the structured answer representation that feeds evaluation."""

    def build(
        self,
        question_id: str,
        text: str,
        diagrams: list[dict[str, Any]],
        equations: list[dict[str, Any]],
        quality: dict[str, Any] | None = None,
        source_paths: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "question_id": question_id,
            "text": text,
            "diagrams": diagrams,
            "equations": equations,
            "quality": quality or {},
            "source_paths": source_paths or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def save(self, path: str | Path, payload: dict[str, Any]) -> str:
        write_json(Path(path), payload)
        return str(path)
