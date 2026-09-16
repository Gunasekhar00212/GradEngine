"""Question splitting helpers for auto mode and manual teacher mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from backend.app.models.schemas import AnnotationClick, PageBoundary
from backend.app.utils.file_utils import write_json
from backend.app.utils.image_utils import crop_image


class SplitService:
    """Store page boundaries and crop question images from a page."""

    def group_pages_by_question(self, clicks: list[AnnotationClick], num_pages: int) -> list[dict[str, int | None]]:
        """Convert teacher question-start clicks into page spans."""

        ordered_clicks = sorted(clicks, key=lambda click: click.question_index)
        groups: list[dict[str, int | None]] = []

        for index, click in enumerate(ordered_clicks):
            if index + 1 < len(ordered_clicks):
                next_click = ordered_clicks[index + 1]
                end_page = next_click.page_index
                end_y: int | None = next_click.y
            else:
                end_page = max(0, num_pages - 1)
                end_y = None

            groups.append(
                {
                    "question_index": click.question_index,
                    "start_page": click.page_index,
                    "start_y": click.y,
                    "end_page": end_page,
                    "end_y": end_y,
                }
            )

        return groups

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

    def load_manual_clicks(self, annotation_path: str | Path) -> list[AnnotationClick]:
        """Load teacher question-start clicks from disk."""

        path = Path(annotation_path)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [AnnotationClick(**item) for item in data]

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

    def crop_question_image(
        self,
        group: dict[str, int | None],
        page_paths: list[str],
        output_dir: str | Path,
    ) -> str:
        """Crop and merge all slices that belong to one question."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        start_page = int(group["start_page"] or 0)
        end_page = int(group["end_page"] or start_page)
        end_y = group["end_y"]
        slices: list[Image.Image] = []

        for page_index in range(start_page, end_page + 1):
            with Image.open(page_paths[page_index]) as image:
                rgb_image = image.convert("RGB")
                width, height = rgb_image.size
                top = int(group["start_y"] or 0) if page_index == start_page else 0
                bottom = int(end_y) if page_index == end_page and end_y is not None else height
                if bottom <= top:
                    bottom = min(height, top + 1)
                slices.append(rgb_image.crop((0, top, width, bottom)))

        target = output / f"q{int(group['question_index'] or 0)}.png"
        if not slices:
            return str(target)
        if len(slices) == 1:
            slices[0].save(target)
            return str(target)

        merged_width = max(image.width for image in slices)
        merged_height = sum(image.height for image in slices)
        canvas = Image.new("RGB", (merged_width, merged_height), "white")
        y_offset = 0
        for slice_image in slices:
            canvas.paste(slice_image, (0, y_offset))
            y_offset += slice_image.height
        canvas.save(target)
        return str(target)

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

    def validate_boundaries(self, groups: list[dict[str, int | None]], num_pages: int) -> dict[str, Any]:
        """Validate question boundary groups for obvious errors.
        
        Returns:
        {
            "valid": bool,
            "errors": list[str],
            "warnings": list[str]
        }
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not groups:
            errors.append("No questions detected")
            return {"valid": False, "errors": errors, "warnings": warnings}

        for i, group in enumerate(groups):
            q_index = i + 1
            start_page = int(group.get("start_page") or 0)
            end_page = int(group.get("end_page") or start_page)
            start_y = int(group.get("start_y") or 0)
            end_y = group.get("end_y")

            # Basic range checks
            if start_page < 0 or start_page >= num_pages:
                errors.append(f"Q{q_index}: start_page {start_page} out of range [0, {num_pages-1}]")
            if end_page < 0 or end_page >= num_pages:
                errors.append(f"Q{q_index}: end_page {end_page} out of range [0, {num_pages-1}]")
            if start_page > end_page:
                errors.append(f"Q{q_index}: starts on page {start_page} but ends on page {end_page}")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def detect_likely_continuation(self, text: str) -> bool:
        """Heuristically detect if OCR text looks like a sentence continuation."""
        if not text:
            return False

        text = text.strip()
        first_line = text.split("\n")[0] if "\n" in text else text

        # Starts with lowercase (likely not a new sentence)
        if first_line and first_line[0].islower():
            return True

        # Starts with continuation phrases
        continuation_phrases = [
            "a ", "an ", "the ", "and ", "or ", "but ", "because", "which", "that", "this",
            "these", "those", "it ", "its ", "then ", "when ", "where ", "who ", "what ", "why ",
            "how ", "if ", "unless ", "while ", "after ", "before ", "during ", "within ",
            "he ", "she ", "they ", "we ", "you ", "him ", "her ", "them ", "us "
        ]

        for phrase in continuation_phrases:
            if first_line.lower().startswith(phrase):
                return True

        return False

