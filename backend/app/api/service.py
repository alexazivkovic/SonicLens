"""Upload handling and the analysis pipeline behind POST /analyze.

Audio is never stored: an upload is streamed to a temp file (hashing it on the way), analysed,
and the file is deleted as soon as analysis ends, whether it succeeded or failed. A cache hit
deletes it before any decoding.
"""

import hashlib
import os
import tempfile
import time
from pathlib import Path

from fastapi import UploadFile

from app.core.audio import load_audio
from app.core.version import ESSENTIA_VERSION, EXTRACTOR_VERSION
from app.extraction import extract_features
from app.model.predictor import PerceptualModel

CHUNK_BYTES = 1024 * 1024


class UploadTooLargeError(Exception):
    pass


async def save_upload(upload: UploadFile, temp_dir: Path, suffix: str, max_bytes: int) -> tuple[Path, str]:
    """Stream an upload to a new temp file and hash it. Returns (path, sha256 hex).
    The caller owns the file and must delete it; on any error here it is deleted already."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=temp_dir, suffix=suffix, prefix="upload-")
    path = Path(name)
    digest, size = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := await upload.read(CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLargeError
                digest.update(chunk)
                f.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return path, digest.hexdigest()


def analyse_file(path: Path, model: PerceptualModel) -> dict:
    """Decode, extract and predict. Blocking; the API runs it in a worker thread.
    Returns the descriptor payload that is stored in the cache and returned by the API."""
    audio, sr = load_audio(path)
    result = extract_features(audio, sr)
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "essentia_version": ESSENTIA_VERSION,
        "model_version": model.model_version,
        "musical": result["musical"],
        "perceptual": model.predict(result["model_input"]["features"]),
        "research": result["research"],
        "signal": result["signal"],
        "model_input": result["model_input"],
    }


def elapsed_ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))
