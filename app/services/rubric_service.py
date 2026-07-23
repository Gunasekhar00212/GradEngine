"""Rubric parser and normalization helpers."""

from __future__ import annotations

from pathlib import Path

from app.utils.json_utils import load_json


class RubricService:
    """Convert rubric files into a predictable structure for grading."""

    def parse(self, rubric_path: str | Path) -> dict[str, object]:
        """Load the rubric JSON and fill in safe defaults when fields are missing."""

        rubric = load_json(Path(rubric_path), default={}) or {}
        concepts = rubric.get("concepts", {})
        expected_concepts = list(concepts.keys())
        total_marks = int(sum(item.get("marks", 0) for item in concepts.values()))
        return {
            "question": rubric.get("question", ""),
            "expected_concepts": expected_concepts,
            "concepts": concepts,
            "total_marks": total_marks,
        }
