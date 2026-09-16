"""Pix2TeX equation extraction for cropped equation regions."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class EquationExtractionService:
    """Convert equation images to LaTeX using Pix2TeX's LatexOCR model."""

    _model: Any | None = None

    def extract(self, image_path: str | Path) -> dict[str, str]:
        path = Path(image_path)
        if not path.is_file():
            return {"latex": "", "status": "error", "error": f"Image not found: {path}"}

        try:
            from PIL import Image

            latex = self._get_model()(Image.open(path).convert("RGB")).strip()
            return {"latex": latex, "status": "ok", "error": ""}
        except Exception as exc:
            return {"latex": "", "status": "error", "error": str(exc)}

    def to_latex(self, image_path: str | Path) -> str:
        """Compatibility convenience method for callers that only need LaTeX."""

        return self.extract(image_path)["latex"]

    @classmethod
    def _get_model(cls) -> Any:
        if cls._model is None:
            from pix2tex.cli import LatexOCR

            cls._model = LatexOCR()
        return cls._model
