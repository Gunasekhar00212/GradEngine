"""Format the final evaluation payload and save it to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.models.schemas import EvaluationSummary, QuestionEvaluation, RubricSummary, StudentAnswer
from backend.app.utils.file_utils import write_json


class OutputService:
    """Convert the evaluation result into the JSON structure shown in the UI."""

    def format_question_result(
        self,
        question_id: str,
        student_answer: dict[str, Any],
        rubric: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> QuestionEvaluation:
        """Build a validated response object for one question."""

        answer_model = StudentAnswer(**student_answer)
        rubric_model = RubricSummary(**rubric)
        evaluation_model = EvaluationSummary(**evaluation)
        return QuestionEvaluation(
            question_id=question_id,
            student_answer=answer_model,
            rubric=rubric_model,
            evaluation=evaluation_model,
        )

    def save(self, path: str | Path, payload: dict[str, Any]) -> str:
        """Write the final output JSON to disk."""

        write_json(Path(path), payload)
        return str(path)
