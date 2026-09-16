"""Repository for teacher review documents."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from backend.app.repositories._backend import get_collection_backend


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewRepository:
    """Persist teacher overrides and final marks for later training data."""

    def __init__(self) -> None:
        self._collection = get_collection_backend("teacher_reviews")

    def upsert_review(self, review_id: str, document: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(document)
        payload.setdefault("review_id", review_id)
        payload.setdefault("created_at", _now_iso())
        payload["updated_at"] = _now_iso()
        return self._collection.upsert("review_id", review_id, payload)

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        return self._collection.find_one("review_id", review_id)
