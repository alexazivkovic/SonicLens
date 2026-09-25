import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from app.core.version import EXTRACTOR_VERSION
from app.extraction import MODEL_FEATURE_NAMES
from app.model.predictor import PERCEPTUAL_KEYS


def make_bundle(constant: float = 0.3, **overrides) -> dict:
    """A tiny model bundle in the real artifact format: constant predictors, no FMA data needed."""
    X = np.zeros((2, len(MODEL_FEATURE_NAMES)))
    models = {
        t: DummyRegressor(strategy="constant", constant=constant).fit(X, [constant, constant])
        for t in PERCEPTUAL_KEYS
    }
    bundle = {
        "model_version": "test-0",
        "extractor_version": EXTRACTOR_VERSION,
        "feature_names": list(MODEL_FEATURE_NAMES),
        "targets": list(PERCEPTUAL_KEYS),
        "models": models,
    }
    bundle.update(overrides)
    return bundle


@pytest.fixture
def dummy_model_path(tmp_path):
    path = tmp_path / "dummy_model.joblib"
    joblib.dump(make_bundle(), path)
    return path
