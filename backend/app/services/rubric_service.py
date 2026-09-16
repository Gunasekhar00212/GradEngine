"""Rubric parser and normalization helpers."""

from __future__ import annotations

from pathlib import Path

from backend.app.utils.json_utils import load_json


class RubricService:
    """Convert rubric files into a predictable structure for grading."""

    def parse(self, rubric_path: str | Path) -> dict[str, dict[str, object]]:
        """Load the rubric JSON and normalize it into per-question payloads."""

        rubric = load_json(Path(rubric_path), default={}) or {}
        per_question: dict[str, dict[str, object]] = {}

        questions = rubric.get("questions", [])
        if isinstance(questions, list) and questions:
            for index, question in enumerate(questions, start=1):
                question_id = str(
                    question.get("question_id")
                    or question.get("question_no")
                    or question.get("id")
                    or f"Q{index}"
                )
                criteria = self._normalize_criteria(question.get("criteria", []))
                total_marks = int(question.get("max_marks", 0) or sum(item["max_marks"] for item in criteria))
                per_question[question_id] = {
                    "question_id": question_id,
                    "expected_concepts": [item["concept"] for item in criteria],
                    "total_marks": total_marks,
                    "criteria": criteria,
                }

        if per_question:
            return per_question

        concepts = rubric.get("concepts", {})
        if isinstance(concepts, dict) and concepts:
            criteria = [
                {
                    "concept": str(concept),
                    "max_marks": int((data or {}).get("marks", 1) or 1),
                }
                for concept, data in concepts.items()
            ]
            per_question[rubric.get("question_id", "Q1") or "Q1"] = {
                "question_id": rubric.get("question_id", "Q1") or "Q1",
                "expected_concepts": [item["concept"] for item in criteria],
                "total_marks": int(sum(item["max_marks"] for item in criteria)),
                "criteria": criteria,
            }
            return per_question

        expected_concepts = rubric.get("expected_concepts", []) or []
        criteria = [
            {
                "concept": str(concept),
                "max_marks": int(rubric.get("total_marks", 0) or 0) // max(1, len(expected_concepts)) or 1,
            }
            for concept in expected_concepts
        ]
        per_question[rubric.get("question_id", "Q1") or "Q1"] = {
            "question_id": rubric.get("question_id", "Q1") or "Q1",
            "expected_concepts": [item["concept"] for item in criteria],
            "total_marks": int(rubric.get("total_marks", 0) or 0),
            "criteria": criteria,
        }
        return per_question

    @staticmethod
    def _normalize_criteria(raw_criteria: object) -> list[dict[str, object]]:
        """Convert rubric criteria entries into a stable grading shape."""

        criteria: list[dict[str, object]] = []
        if not isinstance(raw_criteria, list):
            return criteria

        for item in raw_criteria:
            if not isinstance(item, dict):
                continue
            concept = str(
                item.get("description")
                or item.get("concept")
                or item.get("label")
                or item.get("name")
                or "criterion"
            )
            marks = item.get("marks", item.get("max_marks", 1))
            try:
                max_marks = int(marks or 1)
            except (TypeError, ValueError):
                max_marks = 1
            criteria.append({"concept": concept, "max_marks": max_marks})

        return criteria
