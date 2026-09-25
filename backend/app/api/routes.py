"""HTTP endpoints."""

import logging
import time
from pathlib import Path as FilePath

from fastapi import APIRouter, File, HTTPException, Path, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import AnalysisResponse, DescriptorInfo, ErrorResponse, HealthResponse, ModelCard
from app.api.service import UploadTooLargeError, analyse_file, elapsed_ms, save_upload
from app.core.config import ALLOWED_EXTENSIONS
from app.core.errors import AudioDecodeError, AudioError
from app.core.version import EXTRACTOR_VERSION
from app.db import cache
from app.db.models import Analysis
from app.extraction.registry import DESCRIPTORS

router = APIRouter()
log = logging.getLogger("soniclens")

HASH_PATTERN = r"^[0-9a-f]{64}$"


def _response(row: Analysis, cached: bool, start: float) -> AnalysisResponse:
    return AnalysisResponse(
        audio_hash=row.audio_hash,
        pipeline_version=row.pipeline_version,
        cached=cached,
        created_at=row.created_at_utc,
        processing_ms=elapsed_ms(start),
        **row.features,
    )


def _require_model(request: Request):
    state = request.app.state
    if state.model is None:
        raise HTTPException(503, f"The perceptual model is not available: {state.model_error}")
    return state.model


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        413: {"model": ErrorResponse, "description": "File larger than the upload limit"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        422: {"model": ErrorResponse, "description": "Corrupt, too short or silent audio"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Analyse an audio file",
)
async def analyze(request: Request, file: UploadFile = File(description="mp3, wav, flac, ogg or m4a")):
    """Upload an audio file and get all descriptors. Identical files are served from the cache
    (`cached: true`) as long as the pipeline version is unchanged. The audio is deleted right
    after analysis and never stored."""
    start = time.perf_counter()
    settings = request.app.state.settings
    model = _require_model(request)

    suffix = FilePath(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(415, f"Unsupported file type '{suffix or '(none)'}'. "
                                 f"Supported: {', '.join(ALLOWED_EXTENSIONS)}.")
    try:
        path, audio_hash = await save_upload(file, settings.temp_dir, suffix, settings.max_upload_bytes)
    except UploadTooLargeError:
        raise HTTPException(413, f"File is larger than the {settings.max_upload_mb:g} MB limit.") from None

    pipeline_version = request.app.state.pipeline_version
    try:
        with request.app.state.sessions() as session:
            row = cache.lookup(session, audio_hash, pipeline_version)
        if row is not None:
            return _response(row, cached=True, start=start)
        try:
            payload = await run_in_threadpool(analyse_file, path, model)
        except AudioDecodeError as exc:
            log.info("Undecodable upload %s: %s", audio_hash[:12], exc)
            raise HTTPException(422, "The file could not be decoded as audio. It may be corrupt, empty or "
                                     "not really in the format its extension suggests.") from None
        except AudioError as exc:
            raise HTTPException(422, str(exc)) from None
    finally:
        path.unlink(missing_ok=True)

    with request.app.state.sessions() as session:
        row, inserted = cache.insert(session, audio_hash, pipeline_version, payload)
    return _response(row, cached=not inserted, start=start)


@router.get(
    "/analyze/{audio_hash}",
    response_model=AnalysisResponse,
    responses={404: {"model": ErrorResponse, "description": "No result for this hash and pipeline version"}},
    summary="Get a stored result by audio hash",
)
async def get_analysis(request: Request, audio_hash: str = Path(pattern=HASH_PATTERN, description="SHA-256 hex")):
    """The stored result for this audio hash under the current pipeline version."""
    start = time.perf_counter()
    if request.app.state.pipeline_version is None:
        _require_model(request)
    with request.app.state.sessions() as session:
        row = cache.lookup(session, audio_hash, request.app.state.pipeline_version)
    if row is None:
        raise HTTPException(404, "No analysis for this audio hash under the current pipeline version.")
    return _response(row, cached=True, start=start)


@router.get("/features", response_model=list[DescriptorInfo], summary="Metadata for every descriptor")
async def features() -> list[DescriptorInfo]:
    """Every descriptor with its group, description, unit and range, method (`dsp` or
    `model`), tier (`core`/`advanced`) and whether it is estimated."""
    return [DescriptorInfo(**d.to_dict()) for d in DESCRIPTORS]


@router.get(
    "/model",
    response_model=ModelCard,
    responses={503: {"model": ErrorResponse, "description": "Model card not available"}},
    summary="Model card of the perceptual-descriptor model",
)
async def model_card(request: Request) -> ModelCard:
    """Training data, evaluation protocol, cross-validation and test metrics per descriptor,
    and known limitations."""
    card = request.app.state.model_card
    if card is None:
        raise HTTPException(503, "The model card is not available.")
    return card


@router.get("/health", response_model=HealthResponse, summary="Liveness and component status")
async def health(request: Request) -> HealthResponse:
    state = request.app.state
    ok = state.essentia_loaded and state.model is not None
    return HealthResponse(
        status="ok" if ok else "degraded",
        essentia_loaded=state.essentia_loaded,
        essentia_version=state.essentia_version,
        model_loaded=state.model is not None,
        model_version=state.model.model_version if state.model else None,
        model_error=state.model_error,
        extractor_version=EXTRACTOR_VERSION,
        pipeline_version=state.pipeline_version,
    )
