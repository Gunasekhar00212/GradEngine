"""Pydantic models that describe the GradEngine prototype payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
	"""Return the ids and file names after a teacher uploads files."""

	session_id: str
	student_file: str
	rubric_file: str
	mode: Literal["auto", "manual"]


class PageBoundary(BaseModel):
	"""Store the start and end positions of one question crop."""

	page_index: int
	start_y: int
	end_y: int
	source: Literal["auto", "manual"] = "auto"


class AnnotationClick(BaseModel):
	"""Store a teacher click on a page image."""

	page_index: int
	y: int
	question_index: int


class DiagramItem(BaseModel):
	"""Describe one diagram region in the output JSON."""

	image_path: str
	labels: list[str] = Field(default_factory=list)


class EquationItem(BaseModel):
	"""Describe one equation region in the output JSON."""

	image_path: str
	latex: str = ""


class StudentAnswer(BaseModel):
	"""Group the extracted text, diagrams, and equations for one answer."""

	text: str = ""
	diagrams: list[DiagramItem] = Field(default_factory=list)
	equations: list[EquationItem] = Field(default_factory=list)
	quality: dict[str, Any] = Field(default_factory=dict)
	source_paths: dict[str, Any] = Field(default_factory=dict)


class RubricSummary(BaseModel):
	"""Hold the rubric concepts and total marks used for grading."""

	expected_concepts: list[str] = Field(default_factory=list)
	total_marks: int = 0
	criteria: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
	"""Hold the grader output in the exact structure the UI shows."""

	marks: float | None = None
	max_marks: float = 0
	feedback: str = ""
	criteria_results: list[dict[str, Any]] = Field(default_factory=list)
	semantic_alignment: float = 0
	rubric_coverage: float = 0
	answer_completeness: float = 0
	equation_correctness: float = 0
	diagram_coverage: float = 0
	evaluation_confidence: float = 0
	needs_human_review: bool = False
	evaluation_source: str = "GEMINI"


class QuestionEvaluation(BaseModel):
	"""The final JSON object for one question."""

	question_id: str
	student_answer: StudentAnswer
	rubric: RubricSummary
	evaluation: EvaluationSummary


class SessionState(BaseModel):
	"""Keep the in-memory session state for the current prototype run."""

	session_id: str
	student_file: str
	rubric_file: str
	mode: Literal["auto", "manual"]
	upload_path: str
	page_paths: list[str] = Field(default_factory=list)
	boundaries: list[PageBoundary] = Field(default_factory=list)
	question_paths: list[str] = Field(default_factory=list)
	extracted_questions: list[dict[str, Any]] = Field(default_factory=list)
	evaluation: list[QuestionEvaluation] = Field(default_factory=list)
	output_path: str = ""


class StoredOutput(BaseModel):
	"""Wrap a saved evaluation file path for the UI."""

	path: str


class SimpleMessage(BaseModel):
	"""Return a short status message from API routes."""

	message: str
