"""Backend launcher for the GradEngine FastAPI app."""

from __future__ import annotations

import uvicorn

from backend.app.main import app


def main() -> None:
    """Run the development server when the file is executed directly."""

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
