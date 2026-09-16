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

	# DEBUG: Log splitting process
	print("\n" + "="*80)
	print("SPLIT SESSION DEBUGGING")
	print("="*80)
	print(f"Session ID: {session_id}")
	print(f"Mode: {session.mode}")
	print(f"Total pages: {len(session.page_paths)}")
	print(f"Page paths: {session.page_paths}")

	if session.mode == "manual":
		annotation_path = PATHS.upload_dir / session_id / "manual_clicks.json"
		clicks = split_service.load_manual_clicks(annotation_path)
		groups = split_service.group_pages_by_question(clicks, len(session.page_paths))
		print(f"\nManual mode - Groups from clicks:")
		for i, group in enumerate(groups, start=1):
			print(f"  Q{i}: pages {group.get('start_page')} to {group.get('end_page')}, y: {group.get('start_y')} to {group.get('end_y')}")
		for group in groups:
			question_path = split_service.crop_question_image(group, session.page_paths, question_dir)
			question_paths.append(question_path)
			all_boundaries.append(group)
			print(f"  Cropped: {question_path}")
	else:
		print(f"\nAuto mode - Detecting boundaries per page:")
		for page_index, page_path in enumerate(session.page_paths):
			boundaries = split_service.auto_detect_boundaries(page_path, page_index)
			print(f"  Page {page_index}: found {len(boundaries)} questions")
			for i, b in enumerate(boundaries, start=1):
				print(f"    Q{i}: y {b.start_y} to {b.end_y}")
			question_paths.extend(split_service.crop_questions(page_path, boundaries, question_dir, page_index))
			session_boundaries.extend(boundaries)
			all_boundaries.extend([boundary.model_dump() for boundary in boundaries])

	print(f"\nTotal questions detected: {len(question_paths)}")
	print(f"Question paths: {question_paths}")
	print("="*80 + "\n")

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


@router.get("/api/sessions/{session_id}/split-preview")
def get_split_preview(session_id: str) -> dict[str, Any]:
	"""Get crop previews and OCR snippets to verify the split was correct."""

	session = _get_session_or_404(session_id)
	if not session.question_paths or not session.boundaries:
		raise HTTPException(status_code=400, detail="Split the pages first")

	# Validate the boundaries first
	validation = split_service.validate_boundaries(session.boundaries, len(session.page_paths))
	
	previews = []
	for index, question_path in enumerate(session.question_paths, start=1):
		question_id = f"Q{index}"
		boundary = session.boundaries[index - 1] if index - 1 < len(session.boundaries) else {}
		
		# Quick OCR preview of the crop
		region_dir = PATHS.extracted_dir / session_id / question_id / "regions"
		layout = layout_service.create_region_crops(question_path, region_dir)
		ocr_results = [ocr_service.extract(region["image_path"]) for region in layout["text_regions"]]
		ocr_text = " ".join([result["text"] for result in ocr_results if result["text"]])
		
		# Use the new detection method
		looks_like_continuation = split_service.detect_likely_continuation(ocr_text)
		
		previews.append({
			"question_id": question_id,
			"boundary": boundary,
			"crop_path": f"/api/sessions/{session_id}/crop/{question_id}",
			"ocr_start": ocr_text[:200],
			"looks_like_continuation": looks_like_continuation,
			"warning": "⚠️ Possible split error: text appears to start mid-sentence" if looks_like_continuation else None,
		})
	
	return {
		"session_id": session_id,
		"validation": validation,
		"previews": previews,
	}


@router.post("/api/sessions/{session_id}/adjust-boundary")
def adjust_boundary(session_id: str, adjustments: dict[str, Any]) -> dict[str, Any]:
	"""Re-split with adjusted boundary coordinates.
	
	Expected format:
	{
		"boundaries": [
			{"start_page": 1, "start_y": 800, "end_page": 4, "end_y": 1500},
			...
		]
	}
	"""

	session = _get_session_or_404(session_id)
	if not session.page_paths:
		raise HTTPException(status_code=400, detail="No page images available")

	question_dir = PATHS.question_crops_dir / session_id
	new_boundaries = adjustments.get("boundaries", [])
	
	if not new_boundaries:
		raise HTTPException(status_code=400, detail="No boundary adjustments provided")

	print("\n" + "="*80)
	print("ADJUST BOUNDARY DEBUGGING")
	print("="*80)
	print(f"Session ID: {session_id}")
	print(f"Adjusting {len(new_boundaries)} questions:")

	question_paths = []
	all_boundaries = []
	
	for i, boundary in enumerate(new_boundaries, start=1):
		print(f"  Q{i}: pages {boundary.get('start_page')} to {boundary.get('end_page')}, y: {boundary.get('start_y')} to {boundary.get('end_y')}")
		question_path = split_service.crop_question_image(boundary, session.page_paths, question_dir)
		question_paths.append(question_path)
		all_boundaries.append(boundary)
		print(f"       Cropped: {question_path}")

	print("="*80 + "\n")

	session.boundaries = session_boundaries = all_boundaries
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
		"message": "Boundaries adjusted and re-cropped. Call /split-preview to verify.",
	}


@router.get("/api/sessions/{session_id}/crop/{question_id}")
def get_crop_image(session_id: str, question_id: str) -> FileResponse:
	"""Return the crop image for preview in the UI."""

	session = _get_session_or_404(session_id)
	
	# Extract question index from question_id (e.g., "Q1" -> 0)
	try:
		q_index = int(question_id[1:]) - 1
	except (ValueError, IndexError):
		raise HTTPException(status_code=400, detail="Invalid question_id format")
	
	if q_index < 0 or q_index >= len(session.question_paths):
		raise HTTPException(status_code=404, detail="Question not found")
	
	crop_path = Path(session.question_paths[q_index])
	if not crop_path.is_file():
		raise HTTPException(status_code=404, detail="Crop image not found")
	
	return FileResponse(crop_path, media_type="image/png")


@router.post("/api/sessions/{session_id}/extract")
def extract_session(session_id: str) -> dict[str, Any]:
	"""Run OCR and region extraction over the split question images."""

	session = _get_session_or_404(session_id)
	if not session.question_paths:
		raise HTTPException(status_code=400, detail="Split the pages first")

	extracted: list[dict[str, Any]] = []
	answer_json_paths: list[str] = []
	
	# DEBUG: Log page-to-question mapping
	print("\n" + "="*80)
	print("EXTRACT SESSION DEBUGGING")
	print("="*80)
	print(f"Session ID: {session_id}")
	print(f"Number of question paths: {len(session.question_paths)}")
	if hasattr(session, 'boundaries') and session.boundaries:
		print(f"Boundaries: {session.boundaries}")
	
	for index, question_path in enumerate(session.question_paths, start=1):
		question_id = f"Q{index}"
		region_dir = PATHS.extracted_dir / session_id / question_id / "regions"
		layout = layout_service.create_region_crops(question_path, region_dir)
		ocr_results = [ocr_service.extract(region["image_path"]) for region in layout["text_regions"]]
		ocr_text = " ".join([result["text"] for result in ocr_results if result["text"]])
		
		# DEBUG: Log what OCR extracted for this question
		print(f"\n--- {question_id} ---")
		print(f"Question Image Path: {question_path}")
		print(f"OCR Text (first 200 chars): {ocr_text[:200]}")
		print(f"OCR Confidence: {[result.get('confidence', 'N/A') for result in ocr_results]}")
		
		answer_json = answer_json_service.build(
			question_id=question_id,
			text=ocr_text,
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
	
	print("="*80 + "\n")

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
		evaluation = evaluation_service.evaluate(extracted, rubric_json, image_path=extracted.get("source_paths", {}).get("question_image"))
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
