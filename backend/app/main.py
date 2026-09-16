"""FastAPI application entrypoint for the GradEngine backend package."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.core.config import PATHS, ensure_workspace_dirs


def create_app() -> FastAPI:
	"""Create the API app and register routes."""

	ensure_workspace_dirs()
	app = FastAPI(title="GradEngine Prototype", version="0.1.0")
	assets_dir = PATHS.frontend_dist_dir / "assets"
	if assets_dir.is_dir():
		app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")
	app.include_router(router)
	return app


app = create_app()

__all__ = ["app", "create_app"]
