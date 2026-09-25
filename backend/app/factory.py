"""Builds the FastAPI application. The runnable instance is app.main:app."""

import json
import logging

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.api.schemas import ModelCard
from app.core.config import Settings
from app.core.version import ESSENTIA_VERSION, pipeline_version
from app.db.cache import make_sessionmaker
from app.model.predictor import load_model

log = logging.getLogger("soniclens")

# Allowance for multipart boundaries and headers on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 64 * 1024


def _check_essentia() -> bool:
    try:
        import essentia.standard as es

        es.Windowing(type="hann")(np.zeros(1024, dtype=np.float32))
        return True
    except Exception:  # noqa: BLE001
        log.exception("Essentia is not usable")
        return False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(
        title="SonicLens",
        version="1.0.0",
        description="Low-level (DSP) and perceptual (learned) audio descriptors. Upload a file to "
                    "`POST /analyze`. Audio is analysed and deleted immediately; only descriptors are stored.",
        license_info={"name": "AGPL-3.0", "url": "https://www.gnu.org/licenses/agpl-3.0.html"},
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    @app.middleware("http")
    async def reject_oversized_uploads(request: Request, call_next):
        # Refuse early, before the body is read, when the declared size is already too large.
        # Uploads without Content-Length are still limited while streaming (save_upload).
        length = request.headers.get("content-length", "")
        if (request.method == "POST" and request.url.path == "/analyze" and length.isdigit()
                and int(length) > settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES):
            return JSONResponse({"detail": f"File is larger than the {settings.max_upload_mb:g} MB limit."},
                                status_code=413)
        return await call_next(request)

    state = app.state
    state.settings = settings
    state.sessions = make_sessionmaker(settings.database_url)
    state.essentia_loaded = _check_essentia()
    state.essentia_version = ESSENTIA_VERSION if state.essentia_loaded else None

    state.model, state.model_error, state.pipeline_version = None, None, None
    try:
        state.model = load_model(settings.model_path)
        state.pipeline_version = pipeline_version(state.model.model_version)
    except Exception as exc:  # noqa: BLE001 - serve /health and /features, report the problem loudly
        state.model_error = f"{type(exc).__name__}: {exc}"
        log.error("Could not load the perceptual model from %s: %s", settings.model_path, state.model_error)

    try:
        state.model_card = ModelCard(**json.loads(settings.model_card_path.read_text()))
    except (OSError, ValueError) as exc:
        state.model_card = None
        log.warning("Model card unavailable (%s): %s", settings.model_card_path, exc)

    app.include_router(router)
    return app

