"""PDF and page-image handling for the prototype."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from app.utils.image_utils import create_placeholder_page


class PdfService:
    """Convert uploaded PDFs into page images or create placeholders when needed."""

    def pdf_to_images(self, pdf_path: str | Path, output_dir: str | Path) -> list[str]:
        """Try to split a PDF into images, then fall back to a placeholder page."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        pdf_path = Path(pdf_path)
        try:
            from pdf2image import convert_from_path

            pages = convert_from_path(str(pdf_path), dpi=220)
            paths: list[str] = []
            for index, page in enumerate(pages, start=1):
                page_path = output / f"page_{index}.png"
                page.save(page_path, "PNG")
                paths.append(str(page_path))
            return paths
        except Exception:
            placeholder = output / f"page_1_{pdf_path.stem}.png"
            return [
                create_placeholder_page(
                    placeholder,
                    f"Placeholder page for {pdf_path.name}",
                    ["Install poppler and pdf2image for real PDF rendering.", "This keeps the prototype runnable."],
                )
            ]

    def copy_pages(self, page_paths: Iterable[str | Path], output_dir: str | Path) -> list[str]:
        """Copy page images into a session folder for editing or annotation."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        copied_paths: list[str] = []
        for index, page_path in enumerate(page_paths, start=1):
            source = Path(page_path)
            target = output / f"page_{index}{source.suffix or '.png'}"
            shutil.copy2(source, target)
            copied_paths.append(str(target))
        return copied_paths
