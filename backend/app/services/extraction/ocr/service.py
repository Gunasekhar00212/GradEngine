"""Gemini Vision OCR for handwritten answer regions."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from backend.app.core.config import PATHS


class OCRService:
    """Transcribe image regions with the configured Gemini multimodal model."""

    _TRANSCRIBE_PROMPT = (
        "Transcribe the text in this exam-answer image exactly. Preserve line breaks where "
        "helpful. Do not explain, correct, grade, or add text. Populate the requested JSON "
        "fields; use unclear_words for words you cannot read confidently."
    )

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key if api_key is not None else PATHS.gemini_api_key
        self._model = model or PATHS.gemini_ocr_model
        self._client: Any | None = None

    def extract(self, image_path: str | Path, prompt: str | None = None) -> dict[str, Any]:
        """Return Gemini text plus a machine-readable extraction status."""

        path = Path(image_path)
        if not path.is_file():
            return {"text": "", "status": "error", "error": f"Image not found: {path}"}
        if not self._api_key:
            return {
                "text": "",
                "status": "not_configured",
                "error": "GEMINI_API_KEY is not configured",
            }

        try:
            client = self._get_client()
            from google.genai import types

            image_bytes, mime_type = self._prepare_image(path)
            response = client.models.generate_content(
                model=self._model,
                contents=[
                    prompt or self._TRANSCRIBE_PROMPT,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "ocr_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "unclear_words": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["text", "ocr_confidence", "unclear_words"],
                    },
                ),
            )
            payload = json.loads(response.text or "{}")
            return {
                "text": str(payload.get("text", "")).strip(),
                "ocr_confidence": max(0.0, min(1.0, float(payload.get("ocr_confidence", 0.0)))),
                "unclear_words": [str(word) for word in payload.get("unclear_words", [])],
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            return {"text": "", "status": "error", "error": str(exc)}

    def extract_text(self, image_path: str | Path) -> str:
        """Compatibility convenience method for callers that only need text."""

        return self.extract(image_path)["text"]

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=PATHS.gemini_ocr_timeout_ms),
            )
        return self._client

    @staticmethod
    def _prepare_image(path: Path) -> tuple[bytes, str]:
        image = Image.open(path)
        image = image.convert("RGB")
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        return buffer.getvalue(), "image/jpeg"

    @staticmethod
    def _mime_type(path: Path) -> str:
        return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
            path.suffix.lower(), "image/png"
        )
