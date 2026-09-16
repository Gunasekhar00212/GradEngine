"""Layout-analysis stage and persistent region-crop creation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


class LayoutAnalysisService:
    """Create deterministic, inspectable crops for each downstream extractor.

    The text-only pipeline gives Gemini the whole question crop to avoid losing multi-line
    handwritten answers. Diagram and equation regions are intentionally omitted until real
    detectors are available.
    """

    def analyze_layout(self, image_path: str | Path) -> dict[str, list[dict[str, Any]]]:
        image = Image.open(image_path)
        width, height = image.size
        return {
            "text_regions": [{"box": [0, 0, width, height], "type": "text"}],
            "diagram_regions": [],
            "equation_regions": [],
        }

    def create_region_crops(
        self, image_path: str | Path, output_dir: str | Path
    ) -> dict[str, list[dict[str, Any]]]:
        """Crop every declared region and attach its durable image path."""

        source_path = Path(image_path)
        layout = self.analyze_layout(source_path)
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(source_path).convert("RGB")
        for region_type, regions in layout.items():
            for index, region in enumerate(regions, start=1):
                left, top, right, bottom = region["box"]
                target = target_dir / f"{source_path.stem}_{region_type}_{index}.png"
                image.crop((left, top, right, bottom)).save(target)
                region["image_path"] = str(target)
        return layout
