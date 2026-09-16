"""Final reporting stage for saving evaluation payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.services.output_service import OutputService


class ReportingService:
    """Wrap the current output writer under the reporting pipeline stage."""

    def __init__(self, output_service: OutputService | None = None) -> None:
        self.output_service = output_service or OutputService()

    def format_question_result(
        self,
        question_id: str,
        student_answer: dict[str, Any],
        rubric: dict[str, Any],
        evaluation: dict[str, Any],
    ):
        return self.output_service.format_question_result(question_id, student_answer, rubric, evaluation)

    def save(self, path: str | Path, payload: dict[str, Any]) -> str:
        return self.output_service.save(path, payload)
