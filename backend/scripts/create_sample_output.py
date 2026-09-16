"""Create a small example output file for the GradEngine prototype."""

from __future__ import annotations

from pathlib import Path

from backend.app.core.config import OUTPUTS_DIR, ensure_workspace_dirs
from backend.app.utils.file_utils import write_json


SAMPLE_OUTPUT = {
    "question_id": "Q1",
    "student_answer": {
        "text": "Plants use sunlight, water, and carbon dioxide to make glucose and oxygen.",
        "diagrams": [{"image_path": "data/question_crops/sample_diagram.png", "labels": ["diagram"]}],
        "equations": [{"image_path": "data/question_crops/sample_equation.png", "latex": "E = mc^2"}],
    },
    "rubric": {
        "expected_concepts": ["sunlight", "CO2", "water", "oxygen", "glucose"],
        "total_marks": 5,
        "criteria": [{"concept": "main concept", "max_marks": 3}],
    },
    "evaluation": {
        "marks": 4.5,
        "max_marks": 5,
        "feedback": "Matched concepts: sunlight, water, oxygen, glucose.",
        "criteria_results": [{"concept": "main concept", "matched": True, "max_marks": 3}],
        "semantic_alignment": 0.9,
        "rubric_coverage": 0.8,
        "answer_completeness": 0.7,
        "equation_correctness": 0.0,
        "diagram_coverage": 0.0,
        "evaluation_confidence": 0.78,
        "needs_human_review": False,
        "evaluation_source": "SAMPLE",
    },
}


def main() -> None:
    """Write the sample output JSON to the prototype output directory."""

    ensure_workspace_dirs()
    output_path = OUTPUTS_DIR / "sample_evaluation.json"
    write_json(output_path, SAMPLE_OUTPUT)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
