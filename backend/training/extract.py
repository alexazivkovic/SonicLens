"""Run the SonicLens extraction over all training clips.

Uses exactly the functions the API uses (app.core.audio.load_audio and
app.extraction.extract_features), in parallel over all CPU cores. Results are appended to
Parquet shards in data/features/, so the run is resumable: already processed or failed track
ids are skipped. Failures are logged to data/extract_failures.jsonl, never raised.

Usage (from backend/):  python -m training.extract [--workers N] [--limit N]
"""

import argparse
import json
import os
import time
import traceback
from multiprocessing import Pool

import pandas as pd

from app.core.audio import load_audio
from app.core.errors import AudioError
from app.core.version import EXTRACTOR_VERSION
from app.extraction import extract_features
from training.common import DATA_DIR, audio_path, load_targets

FEATURES_DIR = DATA_DIR / "features"
FAILURES_PATH = DATA_DIR / "extract_failures.jsonl"
SHARD_SIZE = 500
# FMA clips are the middle 30 s of a track (or the whole track if shorter). A clip clearly
# shorter than that is truncated (see the FMA errata, e.g. mdeff/fma issue #41).
EXPECTED_CLIP_S = 30.0
TRUNCATION_TOLERANCE_S = 1.5


def process(job: tuple[int, float]) -> dict:
    """Extract one clip. Returns a feature row, or a failure record with a `status`."""
    track_id, track_duration_s = job
    try:
        audio, sr = load_audio(audio_path(track_id))
        clip_s = len(audio) / sr
        expected_s = min(EXPECTED_CLIP_S, track_duration_s)
        if clip_s < expected_s - TRUNCATION_TOLERANCE_S:
            return {"track_id": track_id, "status": "truncated",
                    "error": f"clip is {clip_s:.2f} s, expected ~{expected_s:.0f} s"}
        result = extract_features(audio, sr)
    except AudioError as exc:
        return {"track_id": track_id, "status": type(exc).__name__, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - one bad clip must never stop the run
        return {"track_id": track_id, "status": "exception", "error": f"{exc!r}",
                "traceback": traceback.format_exc()}
    return {
        "track_id": track_id,
        "status": "ok",
        "clip_duration_s": clip_s,
        "extractor_version": result["extractor_version"],
        **result["model_input"]["features"],
    }


def load_features() -> pd.DataFrame:
    """All extracted rows, indexed by track_id."""
    shards = sorted(FEATURES_DIR.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No feature shards in {FEATURES_DIR}; run `python -m training.extract`.")
    return pd.concat([pd.read_parquet(p) for p in shards]).set_index("track_id").sort_index()


def _done_ids() -> set[int]:
    done = set()
    for p in FEATURES_DIR.glob("part-*.parquet"):
        done.update(pd.read_parquet(p, columns=["track_id"])["track_id"].tolist())
    if FAILURES_PATH.exists():
        done.update(json.loads(line)["track_id"] for line in FAILURES_PATH.read_text().splitlines())
    return done


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m training.extract", description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=None, help="only process this many clips (for testing)")
    args = parser.parse_args()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    done = _done_ids()
    jobs = [(int(t), float(d)) for t, d in targets["duration_s"].items() if t not in done and audio_path(t).exists()]
    missing_audio = sum(not audio_path(t).exists() for t in targets.index)
    jobs = jobs[: args.limit] if args.limit else jobs
    print(f"{len(targets)} tracks, {len(done)} already processed, {missing_audio} without audio, "
          f"{len(jobs)} to extract with {args.workers} workers (extractor {EXTRACTOR_VERSION})", flush=True)

    shard_index = len(list(FEATURES_DIR.glob("part-*.parquet")))
    rows, n_ok, n_failed = [], 0, 0
    failures = open(FAILURES_PATH, "a")
    start = last_print = time.time()

    def flush() -> None:
        nonlocal rows, shard_index
        if rows:
            pd.DataFrame(rows).to_parquet(FEATURES_DIR / f"part-{shard_index:05d}.parquet", index=False)
            shard_index += 1
            rows = []

    with Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(process, jobs, chunksize=4), start=1):
            if rec["status"] == "ok":
                del rec["status"]
                rows.append(rec)
                n_ok += 1
                if len(rows) >= SHARD_SIZE:
                    flush()
            else:
                n_failed += 1
                failures.write(json.dumps(rec) + "\n")
                failures.flush()
                print(f"  skipped {rec['track_id']}: {rec['error']}", flush=True)
            now = time.time()
            if now - last_print > 30 or i == len(jobs):
                last_print = now
                rate = i / (now - start)
                print(f"  {i}/{len(jobs)} clips ({n_ok} ok, {n_failed} skipped), {rate:.1f} clips/s, "
                      f"ETA {(len(jobs) - i) / rate / 60:.0f} min", flush=True)
    flush()
    failures.close()
    print(f"Done: {n_ok} extracted, {n_failed} skipped (see {FAILURES_PATH}).")


if __name__ == "__main__":
    main()
