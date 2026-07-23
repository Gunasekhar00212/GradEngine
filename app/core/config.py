"""App configuration and workspace paths for the GradEngine prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PAGES_DIR = DATA_DIR / "pages"
CROPS_DIR = DATA_DIR / "crops"
EXTRACTED_DIR = DATA_DIR / "extracted"
OUTPUTS_DIR = DATA_DIR / "outputs"
STATIC_DIR = BASE_DIR / "app" / "web" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "web" / "templates"
DEFAULT_RUBRIC_PATH = DATA_DIR / "rubric" / "expanded_rubric.json"
SAMPLE_OUTPUT_PATH = OUTPUTS_DIR / "sample_evaluation.json"


@dataclass(frozen=True)
class AppPaths:
    """Bundle the main folders so route code can pass them around cleanly."""

    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    upload_dir: Path = UPLOAD_DIR
    pages_dir: Path = PAGES_DIR
    crops_dir: Path = CROPS_DIR
    extracted_dir: Path = EXTRACTED_DIR
    outputs_dir: Path = OUTPUTS_DIR
    static_dir: Path = STATIC_DIR
    templates_dir: Path = TEMPLATES_DIR
    default_rubric_path: Path = DEFAULT_RUBRIC_PATH


PATHS = AppPaths()


def ensure_workspace_dirs() -> None:
    """Create every data folder the prototype needs before it starts."""

    for directory in [
        DATA_DIR,
        UPLOAD_DIR,
        PAGES_DIR,
        CROPS_DIR,
        EXTRACTED_DIR,
        OUTPUTS_DIR,
        STATIC_DIR,
        TEMPLATES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
