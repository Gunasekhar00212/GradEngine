"""Question splitting helpers for auto mode and manual teacher mode."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.models.schemas import AnnotationClick, PageBoundary
from app.utils.file_utils import write_json
from app.utils.image_utils import crop_image


class SplitService:
    """Store page boundaries and crop question images from a page."""

    def auto_detect_boundaries(self, page_path: str | Path, page_index: int = 0) -> list[PageBoundary]:
        """Use a simple whitespace and line-density heuristic to find question starts."""

        image = Image.open(page_path).convert("L")
        width, height = image.size
        pixels = image.load()
        dark_rows: list[int] = []

        for y in range(0, height, 4):
            dark_count = 0
            for x in range(0, width, 8):
                if pixels[x, y] < 210:
                    dark_count += 1
            if dark_count > max(2, width // 120):
                dark_rows.append(y)

        if not dark_rows:
            return [PageBoundary(page_index=page_index, start_y=0, end_y=height, source="auto")]

        boundaries: list[PageBoundary] = []
        start_y = 0
        chunk_start = dark_rows[0]

        for row in dark_rows[1:]:
            if row - chunk_start > 220:
                boundaries.append(PageBoundary(page_index=page_index, start_y=start_y, end_y=chunk_start, source="auto"))
                start_y = chunk_start
            chunk_start = row

        boundaries.append(PageBoundary(page_index=page_index, start_y=start_y, end_y=height, source="auto"))
        return boundaries

    def store_manual_clicks(self, clicks: list[AnnotationClick], output_path: str | Path) -> list[dict[str, int]]:
        """Save the teacher's clicks so the UI can replay them later."""

        payload = [click.model_dump() for click in clicks]
        write_json(Path(output_path), payload)
        return payload

    def load_manual_boundaries(self, annotation_path: str | Path, page_index: int, page_height: int) -> list[PageBoundary]:
        """Convert teacher clicks into page ranges for a simple manual split."""

        path = Path(annotation_path)
        if not path.exists():
            return [PageBoundary(page_index=page_index, start_y=0, end_y=page_height, source="manual")]

        data = json.loads(path.read_text(encoding="utf-8"))
        ys = sorted(int(item["y"]) for item in data if int(item.get("page_index", 0)) == page_index and "y" in item)
        if not ys:
            return [PageBoundary(page_index=page_index, start_y=0, end_y=page_height, source="manual")]

        boundaries: list[PageBoundary] = []
        start_y = 0
        for y in ys:
            if y > start_y:
                boundaries.append(PageBoundary(page_index=page_index, start_y=start_y, end_y=y, source="manual"))
                start_y = y
        boundaries.append(PageBoundary(page_index=page_index, start_y=start_y, end_y=page_height, source="manual"))
        return boundaries

    def crop_questions(
        self,
        page_path: str | Path,
        boundaries: list[PageBoundary],
        output_dir: str | Path,
        page_index: int = 0,
    ) -> list[str]:
        """Crop one image per question range from a page image."""

        page = Path(page_path)
        image = Image.open(page)
        width, height = image.size
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        question_paths: list[str] = []

        for index, boundary in enumerate(boundaries, start=1):
            top = max(0, boundary.start_y)
            bottom = min(height, boundary.end_y)
            if bottom <= top:
                bottom = min(height, top + 1)
            target = output / f"page_{page_index + 1}_q{index}.png"
            question_paths.append(crop_image(page, (0, top, width, bottom), target))
        return question_paths
