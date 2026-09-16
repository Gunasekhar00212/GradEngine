"""PDF/page preprocessing stage for the GradEngine pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from backend.app.services.pdf_service import PdfService


class PreprocessingService:
    """Expose PDF-to-image rendering under the pipeline's preprocessing stage."""

    def __init__(self) -> None:
        self._service = PdfService()

    def render_pages(self, pdf_path: str | Path, output_dir: str | Path) -> list[str]:
        return self._service.pdf_to_images(pdf_path, output_dir)

    def copy_pages(self, page_paths: Iterable[str | Path], output_dir: str | Path) -> list[str]:
        return self._service.copy_pages(page_paths, output_dir)
