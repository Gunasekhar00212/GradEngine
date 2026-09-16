"""Repository for the main evaluation/session document."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from backend.app.repositories._backend import get_collection_backend


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvaluationRepository:
    """Store the main evaluation document in MongoDB or the local fallback."""

    def __init__(self) -> None:
        self._collection = get_collection_backend("evaluations")

    def upsert_session(self, evaluation_id: str, document: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(document)
        payload.setdefault("evaluation_id", evaluation_id)
        payload.setdefault("created_at", _now_iso())
        payload["updated_at"] = _now_iso()
        return self._collection.upsert("evaluation_id", evaluation_id, payload)

    def update_session(self, evaluation_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(updates)
        payload["updated_at"] = _now_iso()
        return self._collection.merge("evaluation_id", evaluation_id, payload)

    def get_session(self, evaluation_id: str) -> dict[str, Any] | None:
        return self._collection.find_one("evaluation_id", evaluation_id)
