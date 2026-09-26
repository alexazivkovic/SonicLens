# SonicLens

SonicLens is a free, open-source audio analysis tool for researchers, developers and music
enthusiasts. Upload an audio file and get a structured set of descriptors:

- **Low-level descriptors** computed with established DSP algorithms from
  [Essentia](https://essentia.upf.edu/): tempo, key and mode, EBU R128 loudness, time signature,
  beat grid, chroma, MFCC, and 28 spectral, timbral, dynamic, rhythmic and tonal statistics.
- **Perceptual descriptors**: energy, danceability, valence, acousticness, instrumentalness,
  liveness and speechiness. They are predicted by a machine-learning model that this project
  trained on low-level descriptors computed from real audio in the
  [FMA dataset](https://github.com/mdeff/fma), with FMA's Echo Nest descriptors as targets.

Audio is analysed and deleted immediately; only descriptors and a hash of the file are stored,
so a repeated upload is answered from the cache.

| Perceptual descriptor | energy | danceability | speechiness | acousticness | valence | instrumentalness | liveness |
|---|---|---|---|---|---|---|---|
| Test R² (unseen artists) | 0.76 | 0.65 | 0.60 | 0.62 | 0.48 | 0.44 | 0.12 |

Each descriptor's reliability is shown next to its value in the UI and served by `GET /model`.

## Documentation

- [`docs/FEATURES.md`](docs/FEATURES.md): every descriptor, with definition, why it is
  included, unit and range, method, whether it is a model input or estimated, and its tier.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): pipeline diagram, train/serve consistency, the
  30 s window, caching and versioning, technology choices.
- [`docs/TRAINING.md`](docs/TRAINING.md): data, splits, models, results, experiments and
  limitations.
- [`backend/training/README.md`](backend/training/README.md): how to run each training step.
- [`backend/reports/SUMMARY.md`](backend/reports/SUMMARY.md): generated evaluation summary and
  plots.

## Installation (macOS)

Requirements: Python 3.11, Node 20+, and ffmpeg (only used to create test audio; Essentia decodes
uploads itself).

```bash
brew install python@3.11 ffmpeg node
```

### Backend

```bash
cd backend
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Essentia on macOS: what worked.** Verified on macOS 26 (Apple Silicon, arm64) with Homebrew
Python 3.11: **the plain pip wheel works**, with no Homebrew tap or Docker needed. Essentia is
pinned to `2.1b6.dev1389` in `requirements.txt`, the newest release that ships a `cp311` macOS
arm64 wheel (later releases only publish wheels for Python 3.14). Essentia's bundled decoder
reads mp3, wav, flac, ogg and m4a directly. Quick check:

```bash
.venv/bin/python -c "import essentia.standard as es; es.MusicExtractor(); print('ok')"
```

The trained model (`backend/models/`, 6 MiB) is committed, so the API works without re-training.

### Client

```bash
cd client
npm install
```

## Running

Start the backend and the client in two terminals:

```bash
cd backend && .venv/bin/uvicorn app.main:app --port 8000
```

```bash
cd client && npm run dev
```

Or use run.sh for starting both in one command:

```bash
./run.sh
```

Use:
```bash
chmod +x run.sh
```
if needed to make it executable.

Open http://localhost:5173. The API docs are at http://127.0.0.1:8000/docs.

| Endpoint | Purpose |
|---|---|
| `POST /analyze` | Upload an audio file (mp3, wav, flac, ogg, m4a; max 50 MB) and get all descriptors |
| `GET /analyze/{audio_hash}` | A stored result for the current pipeline version |
| `GET /features` | Metadata for every descriptor (group, unit, range, method, tier, estimated) |
| `GET /model` | The model card: training data, CV and test metrics, limitations |
| `GET /health` | Whether Essentia and the model are loaded |

A single file can also be analysed from the command line:
`cd backend && .venv/bin/python -m app.extraction path/to/song.mp3` prints the low-level
descriptors and the model input vector as JSON.

**Configuration** (optional environment variables):
- API: `SONICLENS_DATABASE_URL` (default SQLite in `backend/soniclens.db`; any SQLAlchemy URL
  works), `SONICLENS_MAX_UPLOAD_MB` (50), `SONICLENS_TEMP_DIR`, `SONICLENS_CORS_ORIGINS`,
  `SONICLENS_MODEL_PATH`, `SONICLENS_MODEL_CARD_PATH`.
- Client: `VITE_API_URL` (see `client/.env.example`).

## Reproducing the model

The whole pipeline downloads and processes FMA data in `data/` (gitignored). It needs **~13 GiB
of disk** and takes **about 3.5 hours** on a 6-core Apple Silicon Mac:

```bash
cd backend
.venv/bin/python -m training.download metadata   # FMA metadata, SHA-1 verified   ~1.5 min, 1.5 GiB
.venv/bin/python -m training.download targets    # Echo Nest targets + artists     seconds
.venv/bin/python -m training.download audio      # 13,129 clips via range requests ~7.5 min, 11.4 GiB
.venv/bin/python -m training.extract             # SonicLens features, parallel     ~50 min
.venv/bin/python -m training.train               # models + model card              ~53 min
.venv/bin/python -m training.librosa_clips       # comparison features only         ~30 min
.venv/bin/python -m training.evaluate            # experiments, reports, plots      ~25 min
```

Details per step: [`backend/training/README.md`](backend/training/README.md).

## Tests

```bash
cd backend && .venv/bin/pytest
```

62 tests run in ~8 s and need no FMA data (API tests use a tiny dummy model). They cover:
- extraction on synthetic audio, determinism, and train/serve consistency;
- caching and versioning, API schemas and errors;
- that uploads are always deleted.

For the client:

```bash
cd client && npm run build && npm run lint
```

## License

SonicLens is licensed under the **GNU Affero General Public License v3.0** (see `LICENSE`).
This is required because SonicLens links against Essentia, which is itself AGPL-3.0.
`backend/training/librosa_clips.py` adapts code from the FMA repository (MIT License).

## Data attribution

The perceptual-descriptor model is trained on the **Free Music Archive (FMA)** dataset. FMA
metadata is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the audio
tracks are distributed under the licenses chosen by their artists. Training targets are the Echo
Nest audio descriptors that ship inside FMA's metadata. SonicLens's predictions approximate these
descriptors; they are not claimed to match those of any other service.
