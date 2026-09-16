"""Upload ingestion service for evaluation/session bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from backend.app.core.config import PATHS
from backend.app.repositories.evaluation_repository import EvaluationRepository
from backend.app.services.storage.local_storage import LocalStorage
from backend.app.utils.file_utils import make_session_id, slugify


ALLOWED_STUDENT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
ALLOWED_RUBRIC_SUFFIXES = {".json"}


@dataclass(slots=True)
class IngestionResult:
    evaluation_id: str
    mode: str
    files: dict[str, str]
    metadata: dict[str, Any]


class IngestionService:
    """Store uploads and write session metadata without performing grading."""

    def __init__(
        self,
        storage: LocalStorage | None = None,
        evaluation_repository: EvaluationRepository | None = None,
    ) -> None:
        self.storage = storage or LocalStorage(PATHS.data_dir)
        self.evaluation_repository = evaluation_repository or EvaluationRepository()

    def _validate_upload(self, upload: UploadFile, allowed_suffixes: set[str], label: str) -> None:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in allowed_suffixes:
            allowed = ", ".join(sorted(allowed_suffixes))
            raise ValueError(f"{label} must be one of: {allowed}")

    async def ingest_uploads(self, student_file: UploadFile, rubric_file: UploadFile, mode: str = "auto") -> IngestionResult:
        self._validate_upload(student_file, ALLOWED_STUDENT_SUFFIXES, "Student file")
        self._validate_upload(rubric_file, ALLOWED_RUBRIC_SUFFIXES, "Rubric file")

        evaluation_id = make_session_id("evaluation")
        mode_value = mode if mode in {"auto", "manual"} else "auto"
        session_root = f"input/{evaluation_id}"

        student_name = f"answer_sheet{Path(student_file.filename or 'answer_sheet.pdf').suffix.lower()}"
        rubric_name = f"rubric{Path(rubric_file.filename or 'rubric.json').suffix.lower()}"

        student_key = f"{session_root}/{slugify(Path(student_name).stem)}{Path(student_name).suffix.lower()}"
        rubric_key = f"{session_root}/{slugify(Path(rubric_name).stem)}{Path(rubric_name).suffix.lower()}"

        student_path = self.storage.save_file(await student_file.read(), student_key)
        rubric_path = self.storage.save_file(await rubric_file.read(), rubric_key)

        metadata = {
            "evaluation_id": evaluation_id,
            "mode": mode_value,
            "files": {
                "answer_sheet": student_key,
                "rubric": rubric_key,
            },
            "status": "uploaded",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self.evaluation_repository.upsert_session(evaluation_id, metadata)

        return IngestionResult(
            evaluation_id=evaluation_id,
            mode=mode_value,
            files={"answer_sheet": student_path, "rubric": rubric_path},
            metadata=metadata,
        )
