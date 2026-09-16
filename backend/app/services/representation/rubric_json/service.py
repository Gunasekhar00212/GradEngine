"""Convert rubric documents into the structured Rubric JSON format."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.utils.file_utils import write_json


class RubricJsonService:
    """Build a clean rubric structure for the evaluator."""

    def build(self, question_id: str, rubric_payload: dict[str, Any]) -> dict[str, Any]:
        criteria: list[dict[str, Any]] = []
        raw_criteria = rubric_payload.get("criteria") or []
        if raw_criteria:
            for item in raw_criteria:
                criteria.append(
                    {
                        "concept": str(item.get("concept", "criterion")),
                        "max_marks": int(item.get("max_marks", 1) or 1),
                    }
                )
        else:
            concepts = rubric_payload.get("concepts", {})
            if isinstance(concepts, dict) and concepts:
                for concept, data in concepts.items():
                    criteria.append(
                        {
                            "concept": str(concept),
                            "max_marks": int((data or {}).get("marks", 1) or 1),
                        }
                    )
            else:
                expected_concepts = rubric_payload.get("expected_concepts", [])
                total_marks = int(rubric_payload.get("total_marks", 0) or 0)
                default_marks = max(1, total_marks // max(1, len(expected_concepts)))
                for concept in expected_concepts:
                    criteria.append({"concept": str(concept), "max_marks": default_marks})

        return {
            "question_id": question_id,
            "total_marks": int(rubric_payload.get("total_marks", 0) or 0),
            "criteria": criteria,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def save(self, path: str | Path, payload: dict[str, Any]) -> str:
        write_json(Path(path), payload)
        return str(path)
