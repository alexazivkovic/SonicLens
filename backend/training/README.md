# Training pipeline

How to reproduce the SonicLens model. The methodology and results are in
[`docs/TRAINING.md`](../../docs/TRAINING.md).

All commands run from `backend/` with the venv's Python (`.venv/bin/python -m ...`). Data is
written to the repo-level `data/` directory, which is gitignored and holds ~13 GiB when complete.

| Step | Command | Output | Disk | Time (measured) |
|---|---|---|---|---|
| 1. Metadata | `python -m training.download metadata` | `data/fma_metadata.zip` (SHA-1 verified), `data/fma_metadata/*.csv` | 1.5 GiB | ~1.5 min |
| 2. Targets | `python -m training.download targets` | `data/targets.parquet` | < 1 MiB | seconds |
| 3. Audio | `python -m training.download audio` | `data/fma_large/NNN/NNNNNN.mp3` | 11.4 GiB | ~7.5 min |
| 4. Librosa baseline | `python -m training.baseline_librosa` | `reports/baseline_librosa.json` | - | ~2.5 min |
| 5. Extraction | `python -m training.extract` | `data/features/part-*.parquet`, `data/extract_failures.jsonl` | 30 MiB | ~50 min |
| 6. Training | `python -m training.train` | `models/perceptual_model.joblib`, `feature_names.json`, `model_card.json`, `reports/train_results.json` | 6 MiB | ~53 min |
| 7. librosa on clips | `python -m training.librosa_clips` | `data/librosa_clips/part-*.parquet` | 55 MiB | ~30 min |
| 8. Evaluation | `python -m training.evaluate` | `reports/metrics.json`, `reports/SUMMARY.md`, `reports/*.png` | 2 MiB | ~25 min |

Times were measured on a 6-core Apple Silicon Mac (8 GB RAM) with a ~25 MiB/s connection. The
whole pipeline takes about 3.5 hours. Steps 4 and 7 are only needed for the comparison
experiments.

## Notes per script

- **`download.py`**
  - `audio` reads the central directory of `fma_large.zip` over HTTP, then fetches each wanted
    clip with a single range request, decompresses it (bzip2) and verifies its CRC-32. The
    93 GiB archive is never downloaded.
  - It is resumable (clips on disk are skipped). Failures go to `data/download_audio.log`.
  - `--workers` sets the number of concurrent requests (default 16).
- **`extract.py`**
  - Runs the app's own `load_audio` + `extract_features` in one process per CPU core. Results
    are appended to Parquet shards of 500 rows, so an interrupted run resumes where it stopped.
  - Truncated, undecodable, too-short and silent clips are skipped and logged to
    `data/extract_failures.jsonl`.
  - `--limit N` does a trial run.
- **`train.py`**
  - Refuses to run if the feature table was made by a different `EXTRACTOR_VERSION` or lacks
    model inputs.
  - Writes the model bundle, validated by the same code the API uses to load it.
  - Bump `MODEL_VERSION` in `train.py` when retraining, so the API's cache starts fresh rows
    for the new model.
- **`librosa_clips.py`** (evaluation only)
  - A port of FMA's librosa feature code (MIT), re-run on the 30 s clips.
  - It compiles librosa's numba functions once before starting worker processes; concurrent
    first-time compilation corrupted numba's cache and crashed the workers.
- **`evaluate.py`**: CV experiments on the development set, plus diagnostics of the deployed
  model on the test set. `reports/SUMMARY.md` lists key findings computed from the numbers.
- **`baseline_librosa.py`**: the early baseline on FMA's precomputed `features.csv` (CV only).
- **`eval_key_profiles.py`**: the synthetic evaluation that chose the `edma` key profile
  (`reports/key_profiles.json`).
- **`common.py`**: paths, the artist-grouped split, metrics and grouped cross-validation, shared
  by all scripts.

Do not run training and extraction at the same time on a small machine. Gradient boosting's
OpenMP threads competing with the extraction workers slow both to a crawl.
