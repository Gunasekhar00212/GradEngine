"""Backend FastAPI routes for uploads, splitting, grading, and the teacher UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from backend.app.core.config import PATHS, ensure_workspace_dirs
from backend.app.core.state import SessionStateStore, get_session, list_sessions, save_session
from backend.app.models.schemas import AnnotationClick, SimpleMessage, UploadResponse
from backend.app.repositories.evaluation_repository import EvaluationRepository
from backend.app.repositories.review_repository import ReviewRepository
from backend.app.repositories.rubric_repository import RubricRepository
from backend.app.services.evaluation.service import LlmEvaluationService
from backend.app.services.extraction.ocr.service import OCRService
from backend.app.services.ingestion.service import IngestionService
from backend.app.services.layout.service import LayoutAnalysisService
from backend.app.services.output_service import OutputService
from backend.app.services.preprocessing.service import PreprocessingService
from backend.app.services.representation.answer_json.service import AnswerJsonService
from backend.app.services.representation.rubric_json.service import RubricJsonService
from backend.app.services.reporting.service import ReportingService
from backend.app.services.rubric_service import RubricService
from backend.app.services.splitting.service import SplittingService
from backend.app.utils.file_utils import make_session_id, read_json, slugify, write_json

router = APIRouter()

ingestion_service = IngestionService()
preprocessing_service = PreprocessingService()
split_service = SplittingService()
layout_service = LayoutAnalysisService()
ocr_service = OCRService()
rubric_service = RubricService()
evaluation_service = LlmEvaluationService()
output_service = OutputService()
reporting_service = ReportingService(output_service)
answer_json_service = AnswerJsonService()
rubric_json_service = RubricJsonService()
evaluation_repository = EvaluationRepository()
rubric_repository = RubricRepository()
review_repository = ReviewRepository()


def _get_session_or_404(session_id: str) -> SessionStateStore:
	"""Load a session and raise a clear error when it does not exist."""

	session = get_session(session_id)
	if session is None:
		raise HTTPException(status_code=404, detail="Session not found")
	return session


def _safe_filename(name: str) -> str:
	"""Return a file name that is safe to store inside the uploads folder."""

	return slugify(Path(name).stem) + Path(name).suffix.lower()


def _load_template() -> str:
	"""Read the HTML dashboard template from disk."""

	template_path = PATHS.frontend_dist_dir / "index.html"
	if not template_path.is_file():
		template_path = PATHS.templates_dir / "index.html"
	return template_path.read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
	"""Render the simple demo page used by the prototype."""

	return HTMLResponse(_load_template())


@router.get("/api/health", response_model=SimpleMessage)
def health() -> SimpleMessage:
	"""Provide a quick status check for the API."""

	return SimpleMessage(message="GradEngine prototype is running")


@router.post("/api/upload", response_model=UploadResponse)
async def upload_files(
	student_file: UploadFile = File(...),
	rubric_file: UploadFile = File(...),
	mode: str = Form("auto"),
) -> UploadResponse:
	"""Save uploaded files and create a working session."""

	ensure_workspace_dirs()
	try:
		ingestion = await ingestion_service.ingest_uploads(student_file, rubric_file, mode)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	session_id = ingestion.evaluation_id
	student_path = Path(ingestion.files["answer_sheet"])
	rubric_path = Path(ingestion.files["rubric"])

	page_dir = PATHS.pages_dir / session_id
	page_paths = preprocessing_service.render_pages(student_path, page_dir)

	state = SessionStateStore(
		session_id=session_id,
		student_file=str(student_path),
		rubric_file=str(rubric_path),
		mode=ingestion.mode,
		upload_path=str(student_path.parent),
		page_paths=page_paths,
	)
	save_session(state)

	evaluation_repository.update_session(
		session_id,
		{
			"status": "pages_rendered",
			"page_paths": page_paths,
			"storage": ingestion.metadata,
		},
	)

	return UploadResponse(
		session_id=session_id,
		student_file=str(student_path),
		rubric_file=str(rubric_path),
		mode=state.mode,
	)


@router.get("/api/sessions")
def sessions() -> list[dict[str, Any]]:
	"""List the sessions that are currently in memory."""

	return [session.to_dict() for session in list_sessions()]


@router.get("/api/sessions/{session_id}")
def get_session_state(session_id: str) -> dict[str, Any]:
	"""Return one saved session for the dashboard."""

	session = _get_session_or_404(session_id)
	return session.to_dict()


@router.get("/api/sessions/{session_id}/pages/{page_index}")
def get_session_page(session_id: str, page_index: int) -> FileResponse:
	"""Return one rendered page for manual split annotation."""

	session = _get_session_or_404(session_id)
	if page_index < 0 or page_index >= len(session.page_paths):
		raise HTTPException(status_code=404, detail="Page not found")

	page_path = Path(session.page_paths[page_index])
	if not page_path.is_file():
		raise HTTPException(status_code=404, detail="Rendered page file not found")
	return FileResponse(page_path, media_type="image/png")


@router.post("/api/sessions/{session_id}/manual-clicks")
def save_manual_clicks(session_id: str, clicks: list[AnnotationClick]) -> dict[str, Any]:
	"""Store teacher click annotations in a JSON file."""

	session = _get_session_or_404(session_id)
	annotation_path = PATHS.upload_dir / session_id / "manual_clicks.json"
	payload = split_service.store_manual_clicks(clicks, annotation_path)
	save_session(session)
	return {"session_id": session_id, "annotation_path": str(annotation_path), "clicks": payload}


@router.post("/api/sessions/{session_id}/split")
def split_session(session_id: str) -> dict[str, Any]:
	"""Split the uploaded page images into question crops."""

	session = _get_session_or_404(session_id)
	if not session.page_paths:
		raise HTTPException(status_code=400, detail="No page images available")

	question_dir = PATHS.question_crops_dir / session_id
	all_boundaries: list[dict[str, Any]] = []
	question_paths: list[str] = []
	session_boundaries = []

	if session.mode == "manual":
		annotation_path = PATHS.upload_dir / session_id / "manual_clicks.json"
		clicks = split_service.load_manual_clicks(annotation_path)
		groups = split_service.group_pages_by_question(clicks, len(session.page_paths))
		for group in groups:
			question_paths.append(split_service.crop_question_image(group, session.page_paths, question_dir))
			all_boundaries.append(group)
	else:
		for page_index, page_path in enumerate(session.page_paths):
			boundaries = split_service.auto_detect_boundaries(page_path, page_index)
			question_paths.extend(split_service.crop_questions(page_path, boundaries, question_dir, page_index))
			session_boundaries.extend(boundaries)
			all_boundaries.extend([boundary.model_dump() for boundary in boundaries])

	session.boundaries = session_boundaries
	session.question_paths = question_paths
	save_session(session)
	evaluation_repository.update_session(
		session_id,
		{
			"status": "split_complete",
			"boundaries": all_boundaries,
			"question_paths": question_paths,
		},
	)

	return {
		"session_id": session_id,
		"boundaries": all_boundaries,
		"question_paths": question_paths,
	}


@router.post("/api/sessions/{session_id}/extract")
def extract_session(session_id: str) -> dict[str, Any]:
	"""Run OCR and region extraction over the split question images."""

	session = _get_session_or_404(session_id)
	if not session.question_paths:
		raise HTTPException(status_code=400, detail="Split the pages first")

	extracted: list[dict[str, Any]] = []
	answer_json_paths: list[str] = []
	for index, question_path in enumerate(session.question_paths, start=1):
		question_id = f"Q{index}"
		region_dir = PATHS.extracted_dir / session_id / question_id / "regions"
		layout = layout_service.create_region_crops(question_path, region_dir)
		ocr_results = [ocr_service.extract(region["image_path"]) for region in layout["text_regions"]]
		answer_json = answer_json_service.build(
			question_id=question_id,
			text=" ".join([result["text"] for result in ocr_results if result["text"]]),
			diagrams=[],
			equations=[],
			quality={
				"ocr": ocr_results,
				"pipeline_mode": "text_only",
			},
			source_paths={"question_image": question_path, "layout": layout},
		)
		answer_json_path = PATHS.answer_json_dir / session_id / f"{question_id}.json"
		answer_json_service.save(answer_json_path, answer_json)
		answer_json_paths.append(str(answer_json_path))
		extracted.append(answer_json)

	session.extracted_questions = extracted
	session.answer_json_paths = answer_json_paths
	save_session(session)
	evaluation_repository.update_session(
		session_id,
		{
			"status": "extracted_complete",
			"answer_json_paths": answer_json_paths,
			"questions": extracted,
		},
	)
	return {"session_id": session_id, "extracted": extracted}


@router.post("/api/sessions/{session_id}/grade")
def grade_session(session_id: str) -> dict[str, Any]:
	"""Combine the extracted answer with the rubric and produce a score."""

	session = _get_session_or_404(session_id)
	rubric_by_question = rubric_service.parse(session.rubric_file)
	if not session.extracted_questions:
		raise HTTPException(status_code=400, detail="Extract text before grading")

	results = []
	rubric_json_paths: list[str] = []
	for extracted in session.extracted_questions:
		rubric_payload = rubric_by_question.get(
			extracted["question_id"],
			{"question_id": extracted["question_id"], "expected_concepts": [], "total_marks": 0, "criteria": []},
		)
		rubric_json = rubric_json_service.build(extracted["question_id"], rubric_payload)
		rubric_json_path = PATHS.rubric_json_dir / session_id / f"{extracted['question_id']}.json"
		rubric_json_service.save(rubric_json_path, rubric_json)
		rubric_repository.upsert_rubric(f"{session_id}:{extracted['question_id']}", rubric_json)
		rubric_json_paths.append(str(rubric_json_path))
		evaluation = evaluation_service.evaluate(extracted, rubric_json)
		result = reporting_service.format_question_result(
			extracted["question_id"],
			extracted,
			{
				"expected_concepts": [criterion["concept"] for criterion in rubric_json["criteria"]],
				"total_marks": rubric_json["total_marks"],
				"criteria": rubric_json["criteria"],
			},
			evaluation.__dict__,
		)
		results.append(result.model_dump())

	output_payload = results[0] if len(results) == 1 else {"questions": results}
	output_path = PATHS.reports_dir / f"{session_id}_evaluation.json"
	reporting_service.save(output_path, output_payload)
	session.evaluation = results
	session.output_path = str(output_path)
	session.rubric_json_path = rubric_json_paths[0] if rubric_json_paths else ""
	save_session(session)
	evaluation_repository.update_session(
		session_id,
		{
			"status": "evaluation_complete",
			"rubric_json_paths": rubric_json_paths,
			"output_path": str(output_path),
			"evaluation": results,
		},
	)
	return {"session_id": session_id, "output_path": str(output_path), "evaluation": output_payload}


@router.get("/api/sessions/{session_id}/output")
def get_output(session_id: str) -> dict[str, Any]:
	"""Return the saved evaluation JSON for the selected session."""

	session = _get_session_or_404(session_id)
	if not session.output_path:
		raise HTTPException(status_code=404, detail="No output has been generated yet")
	payload = read_json(Path(session.output_path), default={})
	return {"session_id": session_id, "output": payload, "output_path": session.output_path}


@router.get("/api/sample-output")
def sample_output() -> JSONResponse:
	"""Expose a ready-made sample output for the README and the UI."""

	sample_path = PATHS.reports_dir / "sample_evaluation.json"
	if not sample_path.exists():
		sample_payload = {
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
		write_json(sample_path, sample_payload)
	return JSONResponse(content=read_json(sample_path, default={}))
