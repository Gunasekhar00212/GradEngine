"""Small image helpers for cropping and placeholder rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


def load_image(path: str | Path) -> Image.Image:
    """Open an image from disk."""

    return Image.open(path).convert("RGB")


def crop_image(image_path: str | Path, box: tuple[int, int, int, int], output_path: str | Path) -> str:
    """Crop a rectangle from an image and save it to disk."""

    image = load_image(image_path)
    cropped = image.crop(box)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output)
    return str(output)


def create_placeholder_page(output_path: str | Path, title: str, lines: Iterable[str]) -> str:
    """Create a simple page image when real PDF rendering is not available."""

    canvas = Image.new("RGB", (1400, 1800), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((60, 60), title, fill="black", font=font)
    y = 140
    for line in lines:
        draw.text((60, y), line, fill="black", font=font)
        y += 36
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return str(output)
