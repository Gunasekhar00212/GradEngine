"""Local launcher for the GradEngine FastAPI prototype."""

from __future__ import annotations

import uvicorn

from app.main import app


if __name__ == "__main__":
    """Run the development server when the file is executed directly."""

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
