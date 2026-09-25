"""Loading and inference of the perceptual-descriptor model.

The model artifact is a joblib bundle written by training/train.py. It carries the ordered
feature-name list and the extractor version it was trained with; both are validated at load
time, and any mismatch fails loudly instead of producing silently wrong predictions.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from app.core.version import EXTRACTOR_VERSION
from app.extraction import MODEL_FEATURE_NAMES
from app.extraction.registry import keys_in_group

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "perceptual_model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"
PERCEPTUAL_KEYS = keys_in_group("perceptual")


class ModelMismatchError(RuntimeError):
    """The model artifact does not match the running extraction code."""


@dataclass(frozen=True)
class PerceptualModel:
    model_version: str
    extractor_version: str
    feature_names: list[str]
    models: dict  # target -> fitted regressor with .predict(X)

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        """Predict every perceptual descriptor from one model input vector (name -> value)."""
        missing = [n for n in self.feature_names if n not in features]
        if missing:
            raise ModelMismatchError(f"{len(missing)} model inputs missing, e.g. {missing[:3]}")
        x = np.array([[features[n] for n in self.feature_names]], dtype=np.float64)
        return {t: float(np.clip(self.models[t].predict(x)[0], 0.0, 1.0)) for t in PERCEPTUAL_KEYS}


def validate_bundle(bundle: dict) -> None:
    names = list(bundle.get("feature_names", []))
    if names != MODEL_FEATURE_NAMES:
        extra = sorted(set(names) - set(MODEL_FEATURE_NAMES))
        absent = sorted(set(MODEL_FEATURE_NAMES) - set(names))
        detail = f"extra {extra[:3]}, missing {absent[:3]}" if extra or absent else "same names, different order"
        raise ModelMismatchError(
            f"Model feature list does not match the extractor ({len(names)} vs {len(MODEL_FEATURE_NAMES)} "
            f"features; {detail}). Retrain the model."
        )
    if bundle.get("extractor_version") != EXTRACTOR_VERSION:
        raise ModelMismatchError(
            f"Model was trained with extractor {bundle.get('extractor_version')}, "
            f"running extractor is {EXTRACTOR_VERSION}. Retrain the model."
        )
    absent_targets = set(PERCEPTUAL_KEYS) - set(bundle.get("models", {}))
    if absent_targets:
        raise ModelMismatchError(f"Model bundle has no model for {sorted(absent_targets)}.")


def load_model(path: str | Path = MODEL_PATH) -> PerceptualModel:
    bundle = joblib.load(path)
    validate_bundle(bundle)
    return PerceptualModel(
        model_version=bundle["model_version"],
        extractor_version=bundle["extractor_version"],
        feature_names=list(bundle["feature_names"]),
        models=bundle["models"],
    )
