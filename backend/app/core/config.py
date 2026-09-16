"""App configuration and workspace paths for the GradEngine prototype."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ, getenv
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path: str | Path) -> None:
        """Load simple KEY=VALUE entries when python-dotenv is not installed yet."""

        env_path = Path(path)
        if not env_path.is_file():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key and not key.startswith("#"):
                environ.setdefault(key, value.strip().strip("\"'"))


BASE_DIR = Path(__file__).resolve().parents[3]
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
UPLOAD_DIR = INPUT_DIR
PAGES_DIR = DATA_DIR / "pages"
QUESTION_CROPS_DIR = DATA_DIR / "question_crops"
QUESTION_CROP_DIR = QUESTION_CROPS_DIR
CROPS_DIR = QUESTION_CROPS_DIR
TEXT_DIR = DATA_DIR / "text"
DIAGRAMS_DIR = DATA_DIR / "diagrams"
EQUATIONS_DIR = DATA_DIR / "equations"
ANSWER_JSON_DIR = DATA_DIR / "answer_json"
RUBRIC_JSON_DIR = DATA_DIR / "rubric_json"
EVALUATIONS_DIR = DATA_DIR / "evaluations"
RELIABILITY_DIR = DATA_DIR / "reliability"
REPORTS_DIR = DATA_DIR / "reports"
OUTPUTS_DIR = REPORTS_DIR
EXTRACTED_DIR = TEXT_DIR
STATIC_DIR = BASE_DIR / "frontend" / "static"
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
TEMPLATES_DIR = FRONTEND_DIR
DEFAULT_RUBRIC_PATH = DATA_DIR / "rubric" / "expanded_rubric.json"
SAMPLE_OUTPUT_PATH = REPORTS_DIR / "sample_evaluation.json"
MONGODB_URI = getenv("MONGODB_URI", "").strip()
MONGODB_DATABASE = getenv("MONGODB_DATABASE", "gradengine").strip() or "gradengine"
GEMINI_API_KEY = (getenv("GEMINI_API_KEY") or getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_OCR_MODEL = getenv("GEMINI_OCR_MODEL", "gemini-3.5-flash").strip() or "gemini-3.5-flash"
GEMINI_EVALUATION_MODEL = getenv("GEMINI_EVALUATION_MODEL", GEMINI_OCR_MODEL).strip() or GEMINI_OCR_MODEL
GEMINI_OCR_TIMEOUT_MS = int(getenv("GEMINI_OCR_TIMEOUT_MS", "30000").strip() or "30000")
GEMINI_EVALUATION_TIMEOUT_MS = int(getenv("GEMINI_EVALUATION_TIMEOUT_MS", "60000").strip() or "60000")


@dataclass(frozen=True)
class AppPaths:
	"""Bundle the main folders so route code can pass them around cleanly."""

	base_dir: Path = BASE_DIR
	data_dir: Path = DATA_DIR
	input_dir: Path = INPUT_DIR
	upload_dir: Path = INPUT_DIR
	pages_dir: Path = PAGES_DIR
	question_crops_dir: Path = QUESTION_CROPS_DIR
	crops_dir: Path = QUESTION_CROPS_DIR
	text_dir: Path = TEXT_DIR
	diagrams_dir: Path = DIAGRAMS_DIR
	equations_dir: Path = EQUATIONS_DIR
	answer_json_dir: Path = ANSWER_JSON_DIR
	rubric_json_dir: Path = RUBRIC_JSON_DIR
	evaluations_dir: Path = EVALUATIONS_DIR
	reliability_dir: Path = RELIABILITY_DIR
	reports_dir: Path = REPORTS_DIR
	extracted_dir: Path = TEXT_DIR
	outputs_dir: Path = REPORTS_DIR
	static_dir: Path = STATIC_DIR
	frontend_dist_dir: Path = FRONTEND_DIST_DIR
	templates_dir: Path = TEMPLATES_DIR
	default_rubric_path: Path = DEFAULT_RUBRIC_PATH
	mongodb_uri: str = MONGODB_URI
	mongodb_database: str = MONGODB_DATABASE
	gemini_api_key: str = GEMINI_API_KEY
	gemini_ocr_model: str = GEMINI_OCR_MODEL
	gemini_evaluation_model: str = GEMINI_EVALUATION_MODEL
	gemini_ocr_timeout_ms: int = GEMINI_OCR_TIMEOUT_MS
	gemini_evaluation_timeout_ms: int = GEMINI_EVALUATION_TIMEOUT_MS


PATHS = AppPaths()


def ensure_workspace_dirs() -> None:
	"""Create every data folder the prototype needs before it starts."""

	for directory in [
		DATA_DIR,
		INPUT_DIR,
		PAGES_DIR,
		QUESTION_CROPS_DIR,
		TEXT_DIR,
		DIAGRAMS_DIR,
		EQUATIONS_DIR,
		ANSWER_JSON_DIR,
		RUBRIC_JSON_DIR,
		EVALUATIONS_DIR,
		RELIABILITY_DIR,
		REPORTS_DIR,
		FRONTEND_DIR,
		STATIC_DIR,
		DATA_DIR / "uploads",
		DATA_DIR / "crops",
		DATA_DIR / "extracted",
		DATA_DIR / "extracted_text",
		DATA_DIR / "rubric",
	]:
		directory.mkdir(parents=True, exist_ok=True)
