"""LLM evaluation stub and reliability layer for semantic grading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    """Keep the core evaluation values together while grading."""

    marks: float
    feedback: str
    semantic_alignment: float
    rubric_coverage: float
    answer_completeness: float
    evaluation_confidence: float
    needs_human_review: bool


class EvaluationService:
    """Call an LLM only for semantic grading and keep everything else deterministic."""

    def evaluate(self, answer_payload: dict[str, Any], rubric_payload: dict[str, Any]) -> EvaluationResult:
        """Create a simple prototype score based on overlap and answer length."""

        answer_text = (answer_payload.get("text") or "").strip()
        expected_concepts = rubric_payload.get("expected_concepts", [])
        concepts = rubric_payload.get("concepts", {})
        total_marks = max(1, int(rubric_payload.get("total_marks", 1)))

        answer_lower = answer_text.lower()
        matched = [concept for concept in expected_concepts if concept.lower() in answer_lower]
        rubric_coverage = len(matched) / max(1, len(expected_concepts))
        answer_completeness = min(1.0, len(answer_text.split()) / 40.0)
        semantic_alignment = round((rubric_coverage * 0.7) + (answer_completeness * 0.3), 3)
        evaluation_confidence = round(max(0.1, semantic_alignment - 0.1), 3)
        marks = round(total_marks * semantic_alignment, 1)
        needs_human_review = evaluation_confidence < 0.5 or semantic_alignment < 0.45

        if matched:
            feedback = f"Matched concepts: {', '.join(matched)}."
        else:
            feedback = "No clear rubric concepts were detected in the answer."

        if not answer_text:
            feedback = "The answer text is empty, so the score should be checked by a human reviewer."
            needs_human_review = True

        return EvaluationResult(
            marks=marks,
            feedback=feedback,
            semantic_alignment=semantic_alignment,
            rubric_coverage=round(rubric_coverage, 3),
            answer_completeness=round(answer_completeness, 3),
            evaluation_confidence=evaluation_confidence,
            needs_human_review=needs_human_review,
        )

    def explain_reliability(self, result: EvaluationResult) -> str:
        """Return a plain English reason for the review decision."""

        if result.needs_human_review:
            return "Low confidence or weak rubric alignment, so send this answer to a teacher."
        return "The score is stable enough for a prototype auto-grade decision."
