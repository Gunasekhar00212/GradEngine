"""Teacher review stage placeholder and repository bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.repositories.review_repository import ReviewRepository


@dataclass(slots=True)
class TeacherReview:
    evaluation_id: str
    question_id: str
    final_marks: float | None = None
    override_reason: str = ""
    approved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ReviewService:
    """Persist teacher decisions so they can become future training data."""

    def __init__(self, repository: ReviewRepository | None = None) -> None:
        self.repository = repository or ReviewRepository()

    def save_review(self, review: TeacherReview) -> dict[str, Any]:
        payload = {
            "evaluation_id": review.evaluation_id,
            "question_id": review.question_id,
            "final_marks": review.final_marks,
            "override_reason": review.override_reason,
            "approved": review.approved,
            "metadata": review.metadata,
        }
        review_id = f"{review.evaluation_id}:{review.question_id}"
        return self.repository.upsert_review(review_id, payload)
