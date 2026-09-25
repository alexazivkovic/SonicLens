"""Model artifact loading and validation."""

import joblib
import pytest

from app.extraction import MODEL_FEATURE_NAMES
from app.model.predictor import PERCEPTUAL_KEYS, ModelMismatchError, load_model
from tests.conftest import make_bundle

FEATURES = {name: 0.0 for name in MODEL_FEATURE_NAMES}


def test_load_and_predict(dummy_model_path):
    model = load_model(dummy_model_path)
    assert model.model_version == "test-0"
    prediction = model.predict(FEATURES)
    assert list(prediction) == PERCEPTUAL_KEYS
    assert all(v == pytest.approx(0.3) for v in prediction.values())


def test_predictions_are_clipped_to_unit_range(tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(make_bundle(constant=1.7), path)
    assert set(load_model(path).predict(FEATURES).values()) == {1.0}


@pytest.mark.parametrize(
    "overrides",
    [
        {"feature_names": list(reversed(MODEL_FEATURE_NAMES))},  # same names, wrong order
        {"feature_names": MODEL_FEATURE_NAMES[:-1]},  # a feature missing
        {"feature_names": MODEL_FEATURE_NAMES + ["unknown.mean"]},  # an extra feature
        {"extractor_version": "0.0.0"},
    ],
)
def test_mismatched_artifact_fails_loudly(tmp_path, overrides):
    path = tmp_path / "m.joblib"
    joblib.dump(make_bundle(**overrides), path)
    with pytest.raises(ModelMismatchError):
        load_model(path)


def test_bundle_without_all_targets_fails(tmp_path):
    bundle = make_bundle()
    del bundle["models"]["valence"]
    path = tmp_path / "m.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ModelMismatchError, match="valence"):
        load_model(path)


def test_predict_rejects_incomplete_input(dummy_model_path):
    features = dict(FEATURES)
    del features["tempo"]
    with pytest.raises(ModelMismatchError):
        load_model(dummy_model_path).predict(features)
