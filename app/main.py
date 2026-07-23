"""FastAPI application entrypoint for the GradEngine prototype."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.config import PATHS, ensure_workspace_dirs


def create_app() -> FastAPI:
    """Create the API app and register routes."""

    ensure_workspace_dirs()
    app = FastAPI(title="GradEngine Prototype", version="0.1.0")
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=PATHS.static_dir), name="static")
    Jinja2Templates(directory=str(PATHS.templates_dir))
    return app


app = create_app()
