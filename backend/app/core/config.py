"""Runtime configuration, read from environment variables (all optional)."""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    # Any SQLAlchemy URL; e.g. postgresql+psycopg://user:pass@host/soniclens
    database_url: str = field(
        default_factory=lambda: os.environ.get("SONICLENS_DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'soniclens.db'}")
    )
    model_path: Path = field(
        default_factory=lambda: Path(os.environ.get("SONICLENS_MODEL_PATH", BACKEND_DIR / "models" / "perceptual_model.joblib"))
    )
    model_card_path: Path = field(
        default_factory=lambda: Path(os.environ.get("SONICLENS_MODEL_CARD_PATH", BACKEND_DIR / "models" / "model_card.json"))
    )
    max_upload_mb: float = field(default_factory=lambda: float(os.environ.get("SONICLENS_MAX_UPLOAD_MB", "50")))
    # Uploads live here only while they are analysed.
    temp_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("SONICLENS_TEMP_DIR", Path(tempfile.gettempdir()) / "soniclens"))
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _list(
            os.environ.get("SONICLENS_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
        )
    )

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


ALLOWED_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
