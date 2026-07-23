"""Equation-to-LaTeX placeholder service."""

from __future__ import annotations

from pathlib import Path


class EquationService:
    """Convert equation crops into LaTeX strings when a math OCR tool is available."""

    def to_latex(self, image_path: str | Path) -> str:
        """Return a stubbed LaTeX string so the pipeline stays runnable."""

        path = Path(image_path)
        return f"\\text{{equation from {path.stem}}}"
