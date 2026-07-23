"""Diagram extraction placeholder service."""

from __future__ import annotations

from pathlib import Path


class DiagramService:
    """Keep diagrams as images and attach a simple label list."""

    def describe(self, image_path: str | Path) -> dict[str, object]:
        """Return a small structured record that can later be replaced by vision logic."""

        path = Path(image_path)
        return {"image_path": str(path), "labels": ["diagram"]}
