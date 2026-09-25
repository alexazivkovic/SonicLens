"""Paths, data loading, artist-grouped splits and metrics shared by the training scripts."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
DATA_DIR = REPO_ROOT / "data"
METADATA_DIR = DATA_DIR / "fma_metadata"
AUDIO_DIR = DATA_DIR / "fma_large"
TARGETS_PATH = DATA_DIR / "targets.parquet"
REPORTS_DIR = BACKEND_DIR / "reports"
MODELS_DIR = BACKEND_DIR / "models"

SEED = 42
# The perceptual descriptors the model learns. Echo Nest tempo is kept separately, only to
# validate the DSP tempo estimate.
TARGETS = ["energy", "danceability", "valence", "acousticness", "instrumentalness", "liveness", "speechiness"]

TEST_FRACTION = 0.15
N_FOLDS = 5


def audio_path(track_id: int) -> Path:
    """Location of a clip, mirroring the fma_large layout (e.g. 000/000002.mp3)."""
    return AUDIO_DIR / f"{track_id // 1000:03d}" / f"{track_id:06d}.mp3"


def load_targets() -> pd.DataFrame:
    """Training table from download.py: one row per track, indexed by track_id."""
    return pd.read_parquet(TARGETS_PATH)


def _artist_unit(artist_id: int) -> float:
    """A stable pseudo-random number in [0, 1) per artist, independent of the other rows."""
    digest = hashlib.sha256(f"{SEED}:{artist_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_splits(artist_ids: pd.Series) -> pd.DataFrame:
    """Artist-grouped split: ~15% of artists form the held-out test set, the rest are spread
    over N_FOLDS cross-validation folds. An artist's assignment depends only on its id, so any
    subset of tracks (e.g. only clips whose extraction succeeded) keeps the same split and no
    artist ever appears on both sides.
    """
    u = artist_ids.map(_artist_unit)
    is_test = u < TEST_FRACTION
    fold = ((u - TEST_FRACTION) / (1 - TEST_FRACTION) * N_FOLDS).clip(0, N_FOLDS - 1).astype(int)
    return pd.DataFrame({"is_test": is_test, "fold": fold.where(~is_test, -1)}, index=artist_ids.index)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearmanr(y_true, y_pred).statistic) if np.ptp(y_pred) > 0 else 0.0,
    }


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def load_librosa_features() -> pd.DataFrame:
    """FMA's precomputed librosa features (features.csv), flattened to 'feature.stat.n' columns.
    Used only as a comparison baseline; never deployed."""
    df = pd.read_csv(METADATA_DIR / "features.csv", index_col=0, header=[0, 1, 2])
    df.columns = [".".join(c) for c in df.columns]
    df.index.name = "track_id"
    return df.astype("float32")


def make_models() -> dict:
    """Baselines and the default gradient-boosting model (train.py adds a hyperparameter search)."""
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "mean": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "hgb": HistGradientBoostingRegressor(early_stopping=False, random_state=SEED),
    }


def cross_validate(model, X: pd.DataFrame, y: pd.Series, folds: pd.Series) -> dict:
    """Artist-grouped CV over the precomputed folds. Returns metrics pooled over all
    out-of-fold predictions, plus the per-fold R2 for spread."""
    from sklearn.base import clone

    oof = pd.Series(np.nan, index=y.index)
    fold_r2 = []
    for k in sorted(folds.unique()):
        train, val = folds != k, folds == k
        m = clone(model).fit(X[train], y[train])
        pred = np.clip(m.predict(X[val]), 0.0, 1.0)
        oof[val] = pred
        fold_r2.append(float(r2_score(y[val], pred)))
    return {**regression_metrics(y, oof), "r2_folds": fold_r2}


def tempo_agreement(estimated, reference, tolerance: float = 0.04) -> dict[str, float]:
    """Share of tracks whose estimated tempo is within +-4% of the reference tempo
    (accuracy 1), and also counting double/half tempo errors as correct (accuracy 2)."""
    est, ref = np.asarray(estimated, dtype=float), np.asarray(reference, dtype=float)
    within = [np.abs(est - k * ref) <= tolerance * k * ref for k in (1.0, 2.0, 0.5)]
    return {"accuracy_1": float(np.mean(within[0])), "accuracy_2": float(np.mean(np.any(within, axis=0))),
            "n": int(len(est))}
