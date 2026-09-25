"""Train the perceptual-descriptor models on the extracted SonicLens features.

For each target: a mean predictor and a Ridge regression (standardised inputs) as baselines,
and a HistGradientBoostingRegressor chosen by a random hyperparameter search. Model selection
uses 5-fold artist-grouped cross-validation on the development set only. The held-out
artist-grouped test set is used exactly once, after selection, for the reported numbers.
The deployed models are the ones fitted on the development set, i.e. the models the test
metrics describe.

Writes models/perceptual_model.joblib, models/feature_names.json, models/model_card.json and
reports/train_results.json.  Usage (from backend/):  python -m training.train [--n-search N]
"""

import argparse
import datetime as dt
import json
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.core.version import ESSENTIA_VERSION, EXTRACTOR_VERSION
from app.extraction import MODEL_FEATURE_NAMES
from app.model.predictor import MODEL_CARD_PATH, MODEL_PATH, validate_bundle
from training.common import (
    MODELS_DIR, REPORTS_DIR, SEED, TARGETS, assign_splits, cross_validate, load_targets,
    regression_metrics, tempo_agreement,
)
from training.extract import FAILURES_PATH, load_features

MODEL_VERSION = "1.0.0"
RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0]
SEARCH_SPACE = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [200, 400, 800],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 50, 100],
    "l2_regularization": [0.0, 0.1, 1.0],
    "max_features": [0.5, 1.0],
}

FMA_CITATION = (
    "Defferrard, M., Benzi, K., Vandergheynst, P., Bresson, X. (2017). FMA: A Dataset for Music "
    "Analysis. 18th International Society for Music Information Retrieval Conference (ISMIR). "
    "arXiv:1612.01840"
)
LIMITATIONS = [
    "Echo Nest labels were computed on full tracks, while the model sees 30 s clips (the centre of "
    "the track), so parts of a track outside the window cannot influence predictions.",
    "FMA consists mostly of independent, Creative Commons-licensed music; genre and production "
    "coverage differs from mainstream commercial catalogues, so accuracy on such music is unknown.",
    "Predictions approximate Echo Nest-style descriptors as distributed in FMA; they are not "
    "claimed to match the descriptors of any other service.",
    "Liveness and instrumentalness are noisy targets with skewed distributions; see the per-target "
    "test metrics before relying on them.",
]


def hgb(**params) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(early_stopping=False, random_state=SEED, **params)


def ridge(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def test_metrics(model, X_dev, y_dev, X_test, y_test) -> tuple[object, dict]:
    fitted = model.fit(X_dev.to_numpy(), y_dev.to_numpy())
    pred = np.clip(fitted.predict(X_test.to_numpy()), 0.0, 1.0)
    return fitted, regression_metrics(y_test, pred)


def skipped_by_reason() -> dict[str, int]:
    """Clips that extract.py skipped (truncated, too short, silent, undecodable, ...)."""
    if not FAILURES_PATH.exists():
        return {}
    reasons = [json.loads(line)["status"] for line in FAILURES_PATH.read_text().splitlines()]
    return dict(Counter(reasons))


def load_training_table() -> pd.DataFrame:
    features = load_features()
    versions = features["extractor_version"].unique().tolist()
    if versions != [EXTRACTOR_VERSION]:
        raise SystemExit(f"Features were extracted with {versions}, current extractor is {EXTRACTOR_VERSION}. "
                         "Re-run `python -m training.extract` into an empty data/features/.")
    missing = [n for n in MODEL_FEATURE_NAMES if n not in features.columns]
    if missing:
        raise SystemExit(f"Feature table lacks {len(missing)} model inputs, e.g. {missing[:3]}")
    return load_targets().join(features, how="inner")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m training.train", description=__doc__.splitlines()[0])
    parser.add_argument("--n-search", type=int, default=16, help="random search configurations per target")
    args = parser.parse_args()

    df = load_training_table()
    splits = assign_splits(df["artist_id"])
    dev, test = df[~splits["is_test"]], df[splits["is_test"]]
    folds = splits.loc[dev.index, "fold"]
    X_dev, X_test = dev[MODEL_FEATURE_NAMES], test[MODEL_FEATURE_NAMES]
    print(f"{len(df)} tracks with features: {len(dev)} development ({dev.artist_id.nunique()} artists), "
          f"{len(test)} test ({test.artist_id.nunique()} artists); {len(MODEL_FEATURE_NAMES)} features")

    configs = list(ParameterSampler(SEARCH_SPACE, n_iter=args.n_search, random_state=SEED))
    models, card_targets, results = {}, {}, {}
    for target in TARGETS:
        t0 = time.time()
        y_dev, y_test = dev[target], test[target]

        mean_cv = cross_validate(DummyRegressor(), X_dev, y_dev, folds)
        ridge_cv = {a: cross_validate(ridge(a), X_dev, y_dev, folds) for a in RIDGE_ALPHAS}
        best_alpha = max(ridge_cv, key=lambda a: ridge_cv[a]["r2"])

        search = []
        for cfg in configs:
            search.append({"params": cfg, "cv": cross_validate(hgb(**cfg), X_dev, y_dev, folds)})
        best = max(search, key=lambda s: s["cv"]["r2"])

        # The one and only use of the test set.
        _, mean_test = test_metrics(DummyRegressor(), X_dev, y_dev, X_test, y_test)
        _, ridge_test = test_metrics(ridge(best_alpha), X_dev, y_dev, X_test, y_test)
        models[target], hgb_test = test_metrics(hgb(**best["params"]), X_dev, y_dev, X_test, y_test)

        card_targets[target] = {
            "model": "HistGradientBoostingRegressor",
            "hyperparameters": best["params"],
            "cv": best["cv"],
            "test": hgb_test,
            "baselines": {
                "mean": {"cv": mean_cv, "test": mean_test},
                "ridge": {"alpha": best_alpha, "cv": ridge_cv[best_alpha], "test": ridge_test},
            },
        }
        results[target] = {"ridge_cv_by_alpha": ridge_cv, "search": search}
        print(f"  {target:17s} CV R2 mean {mean_cv['r2']:+.3f} ridge {ridge_cv[best_alpha]['r2']:+.3f} "
              f"hgb {best['cv']['r2']:+.3f} | test R2 hgb {hgb_test['r2']:+.3f} MAE {hgb_test['mae']:.3f} "
              f"rho {hgb_test['spearman']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    tempo = tempo_agreement(df["tempo"], df["echonest_tempo"])
    print(f"DSP tempo vs Echo Nest tempo (all {tempo['n']} clips): accuracy 1 {tempo['accuracy_1']:.3f}, "
          f"accuracy 2 (double/half) {tempo['accuracy_2']:.3f}")

    bundle = {
        "model_version": MODEL_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "feature_names": list(MODEL_FEATURE_NAMES),
        "targets": list(TARGETS),
        "models": models,
    }
    validate_bundle(bundle)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(MODEL_FEATURE_NAMES, indent=1) + "\n")

    n_targets_total = len(load_targets())
    card = {
        "model_version": MODEL_VERSION,
        "created": dt.date.today().isoformat(),
        "extractor_version": EXTRACTOR_VERSION,
        "essentia_version": ESSENTIA_VERSION,
        "sklearn_version": sklearn.__version__,
        "description": "One HistGradientBoostingRegressor per perceptual descriptor, predicting Echo Nest-style "
                       "values in [0, 1] from the SonicLens low-level descriptor vector of a centred 30 s window.",
        "training_data": {
            "dataset": "FMA: A Dataset for Music Analysis (fma_large 30 s clips + fma_metadata)",
            "citation": FMA_CITATION,
            "metadata_license": "CC BY 4.0",
            "targets_source": "Echo Nest audio features distributed in FMA's echonest.csv",
            "n_tracks_with_targets": n_targets_total,
            "n_tracks_used": len(df),
            "n_tracks_without_features": n_targets_total - len(df),
            "extraction_skips_by_reason": skipped_by_reason(),
            "n_dev": len(dev),
            "n_test": len(test),
            "n_artists_dev": int(dev["artist_id"].nunique()),
            "n_artists_test": int(test["artist_id"].nunique()),
        },
        "features": {"n": len(MODEL_FEATURE_NAMES), "names_file": "feature_names.json",
                     "window": "centred 30 s (FMA clip rule: start = duration // 2 - 15)"},
        "evaluation": {
            "cv": "5-fold cross-validation grouped by artist on the development set (model selection)",
            "test": "held-out artist-grouped test set (~15% of artists), used once after model selection",
            "hyperparameter_search": f"{len(configs)} random configurations, best by pooled out-of-fold R2",
            "metrics": ["mae", "rmse", "r2", "spearman"],
        },
        "targets": card_targets,
        "tempo_validation": {
            "description": "SonicLens DSP tempo (RhythmExtractor2013 on the 30 s clip) vs. Echo Nest tempo "
                           "(full track); tolerance 4%, accuracy 2 also accepts double/half tempo.",
            **tempo,
        },
        "limitations": LIMITATIONS,
    }
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2) + "\n")
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "train_results.json").write_text(json.dumps(results, indent=1) + "\n")
    size = MODEL_PATH.stat().st_size / 2**20
    print(f"Saved {MODEL_PATH.name} ({size:.1f} MiB), feature_names.json, model_card.json")


if __name__ == "__main__":
    main()
