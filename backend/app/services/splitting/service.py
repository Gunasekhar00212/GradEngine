"""Question splitting stage for automatic and manual modes."""

from __future__ import annotations

from pathlib import Path

from backend.app.services.split_service import SplitService


class SplittingService(SplitService):
    """Keep the current heuristic splitter but expose it under the pipeline stage."""

    def split_pages(self, page_path: str | Path, page_index: int = 0) -> list:
        return self.auto_detect_boundaries(page_path, page_index)
