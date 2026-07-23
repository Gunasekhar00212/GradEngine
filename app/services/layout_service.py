"""Simple layout analysis placeholder for text, diagram, and equation regions."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class LayoutService:
    """Split one question image into rough content regions for later OCR."""

    def analyze_layout(self, image_path: str | Path) -> dict[str, list[dict[str, object]]]:
        """Return a basic region map that can be replaced by a real detector later."""

        image = Image.open(image_path)
        width, height = image.size
        third = max(1, height // 3)
        return {
            "text_regions": [{"box": [0, 0, width, third], "type": "text"}],
            "diagram_regions": [{"box": [0, third, width, min(height, third * 2)], "type": "diagram"}],
            "equation_regions": [{"box": [0, min(height, third * 2), width, height], "type": "equation"}],
        }
