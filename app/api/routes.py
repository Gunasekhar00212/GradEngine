"""FastAPI routes for uploads, splitting, grading, and the teacher UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from app.core.config import PATHS, ensure_workspace_dirs
from app.core.state import SessionStateStore, get_session, list_sessions, save_session
from app.models.schemas import AnnotationClick, SimpleMessage, UploadResponse
from app.services.diagram_service import DiagramService
from app.services.equation_service import EquationService
from app.services.evaluation_service import EvaluationService
from app.services.layout_service import LayoutService
from app.services.ocr_service import OcrService
from app.services.output_service import OutputService
from app.services.pdf_service import PdfService
from app.services.rubric_service import RubricService
from app.services.split_service import SplitService
from app.utils.file_utils import make_session_id, read_json, slugify, write_json

router = APIRouter()

pdf_service = PdfService()
split_service = SplitService()
layout_service = LayoutService()
ocr_service = OcrService()
equation_service = EquationService()
diagram_service = DiagramService()
rubric_service = RubricService()
evaluation_service = EvaluationService()
output_service = OutputService()


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
    session_id = make_session_id("grad")
    session_dir = PATHS.upload_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    student_name = _safe_filename(student_file.filename or "student.pdf")
    rubric_name = _safe_filename(rubric_file.filename or "rubric.json")
    student_path = session_dir / student_name
    rubric_path = session_dir / rubric_name

    student_path.write_bytes(await student_file.read())
    rubric_path.write_bytes(await rubric_file.read())

    page_dir = PATHS.pages_dir / session_id
    page_paths = pdf_service.pdf_to_images(student_path, page_dir)

    state = SessionStateStore(
        session_id=session_id,
        student_file=str(student_path),
        rubric_file=str(rubric_path),
        mode=mode if mode in {"auto", "manual"} else "auto",
        upload_path=str(session_dir),
        page_paths=page_paths,
    )
    save_session(state)

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

    question_dir = PATHS.crops_dir / session_id
    all_boundaries: list[dict[str, Any]] = []
    session_boundaries = []
    question_paths: list[str] = []

    for page_index, page_path in enumerate(session.page_paths):
        page_image = Path(page_path)
        if session.mode == "manual":
            annotation_path = PATHS.upload_dir / session_id / "manual_clicks.json"
            page_height = 0
            try:
                from PIL import Image

                page_height = Image.open(page_image).size[1]
            except Exception:
                page_height = 1800
            boundaries = split_service.load_manual_boundaries(annotation_path, page_index, page_height)
        else:
            boundaries = split_service.auto_detect_boundaries(page_path, page_index)

        question_paths.extend(split_service.crop_questions(page_path, boundaries, question_dir, page_index))
        session_boundaries.extend(boundaries)
        all_boundaries.extend([boundary.model_dump() for boundary in boundaries])

    session.boundaries = session_boundaries
    session.question_paths = question_paths
    save_session(session)

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
    for index, question_path in enumerate(session.question_paths, start=1):
        layout = layout_service.analyze_layout(question_path)
        text_regions = []
        for region in layout["text_regions"]:
            text_regions.append(ocr_service.extract_text(question_path))
        diagrams = [diagram_service.describe(question_path) for _ in layout["diagram_regions"]]
        equations = [{"image_path": question_path, "latex": equation_service.to_latex(question_path)} for _ in layout["equation_regions"]]
        extracted.append(
            {
                "question_id": f"Q{index}",
                "text": " ".join([text for text in text_regions if text]),
                "diagrams": diagrams,
                "equations": equations,
            }
        )

    session.extracted_questions = extracted
    save_session(session)
    return {"session_id": session_id, "extracted": extracted}


@router.post("/api/sessions/{session_id}/grade")
def grade_session(session_id: str) -> dict[str, Any]:
    """Combine the extracted answer with the rubric and produce a score."""

    session = _get_session_or_404(session_id)
    rubric_payload = rubric_service.parse(session.rubric_file)
    if not session.extracted_questions:
        raise HTTPException(status_code=400, detail="Extract text before grading")

    results = []
    for extracted in session.extracted_questions:
        evaluation = evaluation_service.evaluate(extracted, rubric_payload)
        result = output_service.format_question_result(
            extracted["question_id"],
            {"text": extracted["text"], "diagrams": extracted["diagrams"], "equations": extracted["equations"]},
            {
                "expected_concepts": rubric_payload["expected_concepts"],
                "total_marks": rubric_payload["total_marks"],
            },
            evaluation.__dict__,
        )
        results.append(result.model_dump())

    output_payload = results[0] if len(results) == 1 else {"questions": results}
    output_path = PATHS.outputs_dir / f"{session_id}_evaluation.json"
    output_service.save(output_path, output_payload)
    session.evaluation = results
    session.output_path = str(output_path)
    save_session(session)
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

    sample_path = PATHS.outputs_dir / "sample_evaluation.json"
    if not sample_path.exists():
        sample_payload = {
            "question_id": "Q1",
            "student_answer": {
                "text": "Plants use sunlight, water, and carbon dioxide to make glucose and oxygen.",
                "diagrams": [{"image_path": "data/crops/sample_diagram.png", "labels": ["diagram"]}],
                "equations": [{"image_path": "data/crops/sample_equation.png", "latex": "E = mc^2"}],
            },
            "rubric": {"expected_concepts": ["sunlight", "CO2", "water", "oxygen", "glucose"], "total_marks": 5},
            "evaluation": {
                "marks": 4.5,
                "feedback": "Matched concepts: sunlight, water, oxygen, glucose.",
                "semantic_alignment": 0.9,
                "rubric_coverage": 0.8,
                "answer_completeness": 0.7,
                "evaluation_confidence": 0.78,
                "needs_human_review": False,
            },
        }
        write_json(sample_path, sample_payload)
    return JSONResponse(content=read_json(sample_path, default={}))
