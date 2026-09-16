"""Gemini-based evaluation of Answer JSON against Rubric JSON."""

from __future__ import annotations

import json
import time
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from backend.app.core.config import PATHS


@dataclass
class EvaluationResult:
    """Validated evaluation fields returned by the Gemini grading stage."""

    marks: float | None
    max_marks: float
    feedback: str
    criteria_results: list[dict[str, Any]]
    semantic_alignment: float
    rubric_coverage: float
    answer_completeness: float
    equation_correctness: float
    diagram_coverage: float
    evaluation_confidence: float
    needs_human_review: bool
    evaluation_source: str


class LlmEvaluationService:
    """Produce a schema-validated assessment from one answer and its rubric."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key if api_key is not None else PATHS.gemini_api_key
        self._model = model or PATHS.gemini_evaluation_model
        self._client: Any | None = None

    def evaluate(
        self,
        answer_payload: dict[str, Any],
        rubric_payload: dict[str, Any],
        image_path: str | Path | None = None,
    ) -> EvaluationResult:
        """Send only the structured answer and rubric to Gemini for grading."""

        total_marks = float(rubric_payload.get("total_marks", 0) or 0)
        if not self._api_key:
            return self._review_required(total_marks, "GEMINI_API_KEY is not configured.")

        try:
            from google import genai
            from google.genai import types

            if self._client is None:
                self._client = genai.Client(
                    api_key=self._api_key,
                    http_options=types.HttpOptions(timeout=PATHS.gemini_evaluation_timeout_ms),
                )

            image_part = self._image_part(image_path or self._question_image_path(answer_payload), types)

            for attempt in range(3):
                try:
                    contents: list[Any] = [self._prompt(answer_payload, rubric_payload)]
                    if image_part is not None:
                        contents.append(image_part)
                    response = self._client.models.generate_content(
                        model=self._model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_json_schema=self._response_schema(),
                        ),
                    )
                    return self._to_result(json.loads(response.text or "{}"), total_marks)
                except Exception as exc:
                    if attempt < 2 and self._is_transient_gemini_error(exc):
                        time.sleep(2 ** attempt)
                        continue
                    raise
        except Exception as exc:
            return self._review_required(total_marks, f"Gemini evaluation failed: {exc}")

    @staticmethod
    def _prompt(answer_payload: dict[str, Any], rubric_payload: dict[str, Any]) -> str:
        return (
            "Grade the student answer against the rubric. Use only evidence in the answer. "
            "Award marks criterion by criterion, do not exceed total_marks, and flag human "
            "review if OCR text is missing or unreliable. Return JSON that matches the schema.\n\n"
            f"ANSWER_JSON:\n{json.dumps(answer_payload, ensure_ascii=False)}\n\n"
            f"RUBRIC_JSON:\n{json.dumps(rubric_payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "marks": {"type": "number", "minimum": 0},
                "feedback": {"type": "string"},
                "criteria_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion": {"type": "string"},
                            "awarded_marks": {"type": "number", "minimum": 0},
                            "max_marks": {"type": "number", "minimum": 0},
                            "feedback": {"type": "string"},
                        },
                        "required": ["criterion", "awarded_marks", "max_marks", "feedback"],
                    },
                },
                "semantic_alignment": {"type": "number", "minimum": 0, "maximum": 1},
                "rubric_coverage": {"type": "number", "minimum": 0, "maximum": 1},
                "answer_completeness": {"type": "number", "minimum": 0, "maximum": 1},
                "evaluation_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "needs_human_review": {"type": "boolean"},
            },
            "required": [
                "marks", "feedback", "criteria_results", "semantic_alignment", "rubric_coverage",
                "answer_completeness", "evaluation_confidence", "needs_human_review",
            ],
        }

    @staticmethod
    def _to_result(payload: dict[str, Any], total_marks: float) -> EvaluationResult:
        clamp = lambda value: max(0.0, min(1.0, float(value or 0)))
        return EvaluationResult(
            marks=max(0.0, min(total_marks, float(payload.get("marks", 0) or 0))),
            max_marks=total_marks,
            feedback=str(payload.get("feedback", "")),
            criteria_results=list(payload.get("criteria_results", [])),
            semantic_alignment=clamp(payload.get("semantic_alignment")),
            rubric_coverage=clamp(payload.get("rubric_coverage")),
            answer_completeness=clamp(payload.get("answer_completeness")),
            equation_correctness=0.0,
            diagram_coverage=0.0,
            evaluation_confidence=clamp(payload.get("evaluation_confidence")),
            needs_human_review=bool(payload.get("needs_human_review", False)),
            evaluation_source="GEMINI",
        )

    @staticmethod
    def _review_required(total_marks: float, message: str) -> EvaluationResult:
        return EvaluationResult(
            marks=None, max_marks=total_marks, feedback=message, criteria_results=[],
            semantic_alignment=0.0, rubric_coverage=0.0, answer_completeness=0.0,
            equation_correctness=0.0, diagram_coverage=0.0, evaluation_confidence=0.0,
            needs_human_review=True, evaluation_source="GEMINI / FAILED",
        )

    @staticmethod
    def _is_transient_gemini_error(error: Exception) -> bool:
        message = str(error)
        return "UNAVAILABLE" in message or "503" in message

    @staticmethod
    def _question_image_path(answer_payload: dict[str, Any]) -> Path | None:
        source_paths = answer_payload.get("source_paths") or {}
        question_image = source_paths.get("question_image")
        return Path(question_image) if question_image else None

    @staticmethod
    def _image_part(image_path: Path | None, types: Any) -> Any | None:
        if image_path is None:
            return None
        path = Path(image_path)
        if not path.is_file():
            return None
        image = Image.open(path).convert("RGB")
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")
