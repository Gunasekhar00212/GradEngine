"""OpenCV-based diagram-label extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.services.extraction.ocr.service import OCRService


class DiagramExtractionService:
    """Find text-like label groups with OpenCV and transcribe them through Gemini OCR."""

    def __init__(self, ocr_service: OCRService | None = None) -> None:
        self._ocr = ocr_service or OCRService()

    def extract(self, image_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            return {"image_path": str(path), "labels": [], "status": "error", "error": "Image not found"}

        try:
            source = cv2.imread(str(path))
            if source is None:
                raise ValueError("OpenCV could not read image")
            label_boxes = self._find_label_boxes(source)
            labels: list[str] = []
            label_dir = Path(output_dir)
            label_dir.mkdir(parents=True, exist_ok=True)
            for index, (x, y, width, height) in enumerate(label_boxes[:20], start=1):
                crop_path = label_dir / f"{path.stem}_label_{index}.png"
                cv2.imwrite(str(crop_path), source[y : y + height, x : x + width])
                result = self._ocr.extract(
                    crop_path,
                    "Read this diagram label exactly. Return only the label text; return an empty response if it is not text.",
                )
                if result["status"] == "ok" and result["text"]:
                    labels.append(result["text"])
            return {"image_path": str(path), "labels": list(dict.fromkeys(labels)), "status": "ok", "error": ""}
        except Exception as exc:
            return {"image_path": str(path), "labels": [], "status": "error", "error": str(exc)}

    @staticmethod
    def _find_label_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape
        boxes = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = box_width * box_height
            if 12 <= box_height <= max(20, height // 4) and 20 <= box_width <= width and 300 <= area <= width * height * 0.15:
                boxes.append((x, y, box_width, box_height))
        return sorted(boxes, key=lambda box: (box[1], box[0]))
