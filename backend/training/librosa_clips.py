"""Recompute FMA's librosa features on the 30 s training clips (evaluation only).

FMA's precomputed features.csv was computed by FMA's features.py on each file as a whole. Its
scheduling by track duration indicates full-length tracks. SonicLens only sees 30 s clips, so
comparing against features.csv would mix two effects: the feature set and the analysed audio
span. This script re-runs FMA's feature code on exactly the clips SonicLens uses. Then
evaluate.py can compare librosa vs. Essentia on identical audio, and features.csv vs. these
features isolates the effect of the audio span.

compute_features() is a port of `compute_features` in features.py from
https://github.com/mdeff/fma (MIT License, (c) Michaël Defferrard, Kirell Benzi,
Pierre Vandergheynst, Xavier Bresson), adapted to librosa 0.11: `rmse` is now `rms`, and there
is no warnings-as-errors hack. Output columns match features.csv's flattened names
('feature.statistics.number').

Usage (from backend/):  python -m training.librosa_clips [--workers N] [--limit N]
"""

import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool

import librosa
import numpy as np
import pandas as pd
from scipy import stats

from training.common import DATA_DIR, audio_path, load_targets

OUT_DIR = DATA_DIR / "librosa_clips"
FAILURES_PATH = DATA_DIR / "librosa_clips_failures.jsonl"
SHARD_SIZE = 500

FEATURE_SIZES = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12, tonnetz=6, mfcc=20, rmse=1, zcr=1,
                     spectral_centroid=1, spectral_bandwidth=1, spectral_contrast=7, spectral_rolloff=1)
MOMENTS = ("mean", "std", "skew", "kurtosis", "median", "min", "max")


def compute_features(x: np.ndarray, sr: int) -> dict[str, float]:
    out: dict[str, float] = {}

    def feature_stats(name, values):
        moments = {
            "mean": np.mean(values, axis=1), "std": np.std(values, axis=1),
            "skew": stats.skew(values, axis=1), "kurtosis": stats.kurtosis(values, axis=1),
            "median": np.median(values, axis=1), "min": np.min(values, axis=1), "max": np.max(values, axis=1),
        }
        for moment, vals in moments.items():
            for i, v in enumerate(vals):
                out[f"{name}.{moment}.{i + 1:02d}"] = float(v)

    feature_stats("zcr", librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512))

    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None))
    feature_stats("chroma_cqt", librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7))
    cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats("chroma_cens", cens)
    feature_stats("tonnetz", librosa.feature.tonnetz(chroma=cens))
    del cqt

    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    feature_stats("chroma_stft", librosa.feature.chroma_stft(S=stft**2, n_chroma=12))
    feature_stats("rmse", librosa.feature.rms(S=stft))
    feature_stats("spectral_centroid", librosa.feature.spectral_centroid(S=stft))
    feature_stats("spectral_bandwidth", librosa.feature.spectral_bandwidth(S=stft))
    feature_stats("spectral_contrast", librosa.feature.spectral_contrast(S=stft, n_bands=6))
    feature_stats("spectral_rolloff", librosa.feature.spectral_rolloff(S=stft))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    feature_stats("mfcc", librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20))
    return out


def process(track_id: int) -> dict:
    warnings.filterwarnings("ignore")
    try:
        x, sr = librosa.load(audio_path(track_id), sr=None, mono=True)
        return {"track_id": track_id, "status": "ok", **compute_features(x, sr)}
    except Exception as exc:  # noqa: BLE001 - log and continue
        return {"track_id": track_id, "status": "failed", "error": repr(exc)}


def load_librosa_clip_features() -> pd.DataFrame:
    shards = sorted(OUT_DIR.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No shards in {OUT_DIR}; run `python -m training.librosa_clips`.")
    return pd.concat([pd.read_parquet(p) for p in shards]).set_index("track_id").sort_index().astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m training.librosa_clips", description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    for p in OUT_DIR.glob("part-*.parquet"):
        done.update(pd.read_parquet(p, columns=["track_id"])["track_id"].tolist())
    if FAILURES_PATH.exists():
        done.update(json.loads(line)["track_id"] for line in FAILURES_PATH.read_text().splitlines())
    # Only clips that SonicLens extracted successfully: the comparison uses identical tracks.
    from training.extract import load_features

    ids = [int(t) for t in load_features().index if int(t) not in done]
    ids = ids[: args.limit] if args.limit else ids
    print(f"{len(done)} already processed, {len(ids)} to process with {args.workers} workers", flush=True)

    # librosa's numba functions use an on-disk cache. Compile them once here, so that the worker
    # processes only read a finished cache: concurrent first-time compilation by several
    # workers corrupted the cache and made every worker segfault.
    warnings.filterwarnings("ignore")
    compute_features(np.random.default_rng(0).standard_normal(5 * 44100).astype(np.float32) * 0.1, 44100)

    shard = len(list(OUT_DIR.glob("part-*.parquet")))
    rows, n_failed = [], 0
    failures = open(FAILURES_PATH, "a")
    start = last = time.time()
    with Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(process, ids, chunksize=4), start=1):
            if rec["status"] == "ok":
                del rec["status"]
                rows.append(rec)
                if len(rows) >= SHARD_SIZE:
                    pd.DataFrame(rows).to_parquet(OUT_DIR / f"part-{shard:05d}.parquet", index=False)
                    shard, rows = shard + 1, []
            else:
                n_failed += 1
                failures.write(json.dumps(rec) + "\n")
                failures.flush()
            if time.time() - last > 30 or i == len(ids):
                last = time.time()
                rate = i / (last - start)
                print(f"  {i}/{len(ids)} clips, {n_failed} failed, {rate:.1f} clips/s, "
                      f"ETA {(len(ids) - i) / rate / 60:.0f} min", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(OUT_DIR / f"part-{shard:05d}.parquet", index=False)
    failures.close()
    print(f"Done, {n_failed} failed (see {FAILURES_PATH}).")


if __name__ == "__main__":
    main()
