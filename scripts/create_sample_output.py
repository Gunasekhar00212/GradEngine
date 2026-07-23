"""Create a small example output file for the GradEngine prototype."""

from __future__ import annotations

from pathlib import Path

from app.core.config import OUTPUTS_DIR, ensure_workspace_dirs
from app.utils.file_utils import write_json


SAMPLE_OUTPUT = {
    "question_id": "Q1",
    "student_answer": {
        "text": "Plants use sunlight, water, and carbon dioxide to make glucose and oxygen.",
        "diagrams": [{"image_path": "data/crops/sample_diagram.png", "labels": ["diagram"]}],
        "equations": [{"image_path": "data/crops/sample_equation.png", "latex": "E = mc^2"}],
    },
    "rubric": {"expected_concepts": ["sunlight", "CO2", "water", "oxygen", "glucose"], "total_marks": 5},
    "evaluation": {
        "marks": 4.5,
        "feedback": "Matched concepts: sunlight, water, oxygen, glucose.",
        "semantic_alignment": 0.9,
        "rubric_coverage": 0.8,
        "answer_completeness": 0.7,
        "evaluation_confidence": 0.78,
        "needs_human_review": False,
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
