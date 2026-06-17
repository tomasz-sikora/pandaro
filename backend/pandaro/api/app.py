"""FastAPI application factory + static SPA serving."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from ..logging_setup import configure_logging, get_logger
from .routes import router

log = get_logger("app")


def create_app() -> FastAPI:
    configure_logging(os.environ.get("PANDARO_LOG_LEVEL", "INFO"))
    s = get_settings()
    app = FastAPI(
        title="Pandaro",
        version="0.1.0",
        description="Efemeryczna analiza nagrań rozmów (ASR, diaryzacja, RAG).",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    # Serve the built SPA if present (production); in dev the SPA runs on Vite.
    dist = Path(s.frontend_dist).resolve()
    if dist.is_dir():
        index_file = dist / "index.html"
        assets = dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/")
        async def index() -> FileResponse:  # pragma: no cover - file serving
            return FileResponse(index_file)

        @app.get("/{path:path}")
        async def spa(path: str) -> FileResponse:  # pragma: no cover - SPA fallback
            # Resolve and confine the candidate to ``dist`` to prevent path
            # traversal (e.g. ``../../etc/passwd``); otherwise serve the SPA.
            candidate = (dist / path).resolve()
            if candidate.is_file() and candidate.is_relative_to(dist):
                return FileResponse(candidate)
            return FileResponse(index_file)
    else:
        log.warning("frontend.dist_missing", dist=str(dist))

    log.info("app.ready", port=s.port, llm=s.llm_model)
    return app


app = create_app()
