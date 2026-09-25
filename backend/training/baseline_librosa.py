"""Early baseline: predict the perceptual descriptors from FMA's precomputed librosa features.

Needs no audio. Uses the same targets and artist-grouped folds as the SonicLens models, and
reports cross-validation metrics only (the held-out test set is reserved for evaluate.py).
Writes reports/baseline_librosa.json.  Usage (from backend/):  python -m training.baseline_librosa
"""

import json
import time

from training.common import REPORTS_DIR, TARGETS, assign_splits, cross_validate, load_librosa_features, load_targets, make_models


def main() -> None:
    targets = load_targets()
    features = load_librosa_features()
    df = targets.join(features, how="inner")
    splits = assign_splits(df["artist_id"])
    dev = df[~splits["is_test"]]
    folds = splits.loc[dev.index, "fold"]
    X = dev[features.columns]
    print(f"{len(dev)} development tracks ({len(df) - len(dev)} held out), {X.shape[1]} librosa features")

    report = {"feature_set": "librosa (FMA features.csv)", "n_dev": len(dev), "n_features": X.shape[1], "cv": {}}
    for target in TARGETS:
        report["cv"][target] = {}
        for name, model in make_models().items():
            t = time.time()
            m = cross_validate(model, X, dev[target], folds)
            report["cv"][target][name] = m
            print(f"  {target:17s} {name:6s} R2 {m['r2']:+.3f}  MAE {m['mae']:.3f}  rho {m['spearman']:.3f}  ({time.time() - t:.0f}s)")

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "baseline_librosa.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
