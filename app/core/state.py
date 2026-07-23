"""In-memory session state for the prototype UI and API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import PageBoundary


@dataclass
class SessionStateStore:
    """Keep the latest session payloads in memory during a demo run."""

    session_id: str
    student_file: str
    rubric_file: str
    mode: str
    upload_path: str
    page_paths: list[str] = field(default_factory=list)
    boundaries: list[PageBoundary] = field(default_factory=list)
    question_paths: list[str] = field(default_factory=list)
    extracted_questions: list[dict[str, Any]] = field(default_factory=list)
    evaluation: list[dict[str, Any]] = field(default_factory=list)
    output_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly version of the session state."""

        return {
            "session_id": self.session_id,
            "student_file": self.student_file,
            "rubric_file": self.rubric_file,
            "mode": self.mode,
            "upload_path": self.upload_path,
            "page_paths": self.page_paths,
            "boundaries": [boundary.model_dump() for boundary in self.boundaries],
            "question_paths": self.question_paths,
            "extracted_questions": self.extracted_questions,
            "evaluation": self.evaluation,
            "output_path": self.output_path,
        }


SESSION_CACHE: dict[str, SessionStateStore] = {}


def save_session(state: SessionStateStore) -> None:
    """Store a session in memory so the UI can continue from it."""

    SESSION_CACHE[state.session_id] = state


def get_session(session_id: str) -> SessionStateStore | None:
    """Look up a saved session by id."""

    return SESSION_CACHE.get(session_id)


def list_sessions() -> list[SessionStateStore]:
    """Return all known sessions for the dashboard."""

    return list(SESSION_CACHE.values())
