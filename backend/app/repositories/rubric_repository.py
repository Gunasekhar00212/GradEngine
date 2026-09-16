"""Repository for normalized rubric documents."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from backend.app.repositories._backend import get_collection_backend


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RubricRepository:
    """Persist rubric JSON documents separately from student answers."""

    def __init__(self) -> None:
        self._collection = get_collection_backend("rubrics")

    def upsert_rubric(self, rubric_id: str, document: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(document)
        payload.setdefault("rubric_id", rubric_id)
        payload.setdefault("created_at", _now_iso())
        payload["updated_at"] = _now_iso()
        return self._collection.upsert("rubric_id", rubric_id, payload)

    def get_rubric(self, rubric_id: str) -> dict[str, Any] | None:
        return self._collection.find_one("rubric_id", rubric_id)
