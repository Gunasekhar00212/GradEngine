"""OCR wrapper layer used by the prototype pipeline."""

from __future__ import annotations

from pathlib import Path


class OcrService:
    """Provide a single place to switch OCR engines later."""

    def extract_text(self, image_path: str | Path) -> str:
        """Try Tesseract first and return empty text if OCR is unavailable."""

        try:
            from PIL import Image
            import pytesseract

            image = Image.open(image_path)
            return pytesseract.image_to_string(image).strip()
        except Exception:
            return ""
