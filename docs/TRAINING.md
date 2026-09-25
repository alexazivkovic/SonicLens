# Training and evaluation

How the perceptual-descriptor model was built and evaluated. Commands and runtimes are in
`backend/training/README.md`. Every number here comes from `backend/models/model_card.json` or
`backend/reports/metrics.json`, which the scripts write.

## Data

**Dataset.** [FMA: A Dataset for Music Analysis](https://github.com/mdeff/fma) (Defferrard et
al., ISMIR 2017). Metadata is licensed CC BY 4.0; the audio is under the licenses its artists
chose. No data derived from a commercial streaming service's API is used.

**Targets.** The Echo Nest `audio_features` group in `echonest.csv` provides energy,
danceability, valence, acousticness, instrumentalness, liveness and speechiness, all in 0-1. It
also provides tempo, which is used only to validate the DSP tempo. The Echo Nest values were
computed on **full tracks**. 13,129 FMA tracks by 2,876 artists have them; artist ids come from
`tracks.csv`.

**Audio.** Only the 30 s clips of those 13,129 tracks are fetched from `fma_large.zip`, using
HTTP range requests: the 93 GiB archive is never downloaded (11.4 GiB of clips). Each clip is
CRC-checked.

**Extraction.** `training/extract.py` runs the same `load_audio` + `extract_features` code as
the API. 26 clips are skipped and logged, all matching FMA's known errata:

| Reason | Clips |
|---|---|
| truncated (more than 1.5 s shorter than the expected clip) | 19 |
| undecodable (no audio stream) | 6 |
| silent | 1 |
| **used** | **13,103** |

## Splits

Splits are grouped by artist, so an artist is never on both sides of any split. This matters
because tracks by one artist share production and style, and ungrouped splits inflate scores.

- **Held-out test set:** about 15% of artists: 2,010 tracks by 416 artists. It is used exactly
  once, after model selection, for the reported numbers.
- **Development set:** 11,093 tracks by 2,458 artists, divided into 5 artist-grouped
  cross-validation folds for model selection and all experiments.

Each artist's assignment comes from a seeded hash of its id (`training/common.py`), rather than
from scikit-learn's `GroupKFold`. The guarantee is the same (artist-disjoint folds), but the
assignment of an artist does not depend on which other tracks are present. So the three feature
sets compared below, and any subset (e.g. clips whose extraction failed), keep exactly the same
split. Fixed seeds (42) are used everywhere.

## Models

For each of the 7 targets, on the 257-value SonicLens model input vector:

| Model | Selection |
|---|---|
| Mean predictor | - (baseline) |
| Ridge regression on standardised inputs | alpha from {1, 10, 100, 1000} by grouped CV |
| **HistGradientBoostingRegressor** (deployed) | 16 random configurations (learning rate, iterations, leaf count, min leaf size, L2, feature subsampling) by grouped CV |

Early stopping is disabled. scikit-learn would otherwise enable it above 10,000 samples, with a
random (not artist-grouped) validation split. The deployed models are those fitted on the
development set, so the test metrics describe exactly what ships. Predictions are clipped to
0-1.

## Results (held-out test set)

| Target | **Test R²** | MAE | RMSE | Spearman | CV R² | Ridge test R² | Mean test R² |
|---|---|---|---|---|---|---|---|
| energy | **0.760** | 0.104 | 0.138 | 0.861 | 0.746 | 0.733 | -0.001 |
| danceability | **0.653** | 0.087 | 0.110 | 0.798 | 0.644 | 0.593 | -0.001 |
| valence | **0.481** | 0.163 | 0.203 | 0.700 | 0.417 | 0.408 | 0.000 |
| acousticness | **0.623** | 0.171 | 0.234 | 0.823 | 0.658 | 0.595 | -0.006 |
| instrumentalness | **0.437** | 0.208 | 0.276 | 0.658 | 0.390 | 0.406 | -0.001 |
| liveness | **0.122** | 0.095 | 0.141 | 0.278 | 0.124 | 0.097 | 0.000 |
| speechiness | **0.603** | 0.047 | 0.087 | 0.617 | 0.524 | 0.396 | -0.001 |

Gradient boosting beats Ridge on every target, and both beat the mean predictor everywhere.
Test R² is higher than CV R² for several targets (valence, speechiness). That is a property of this particular test set, not of tuning: nothing
was selected on it. Predictions are compressed towards the middle of the range, which is
typical for regression under noisy labels (`reports/predicted_vs_true.png`).

**DSP tempo check.** SonicLens tempo (30 s clip) agrees with the Echo Nest tempo (full track)
within ±4% on **56.4%** of the 13,103 clips, and on **69.5%** counting double/half tempo. Most
of the remaining errors are 3:2 or 2:3 relations (`reports/tempo_agreement.png`).

## Experiments (`training/evaluate.py`)

All experiments use grouped cross-validation on the development set, with one fixed
gradient-boosting configuration (the one most often selected above). No feature set benefits
from its own tuning. Full results: `reports/SUMMARY.md`, `reports/metrics.json`, plots in
`reports/`.

### 1. Feature-set comparison

FMA ships precomputed librosa features (`features.csv`). Checking FMA's feature code showed they
were computed on **full-length tracks**. This was confirmed by re-running that code on the clips:
on tracks of 30 s or less it reproduces `features.csv` to a median 1.4%, while on tracks of
120 s or more the values differ by ~30%. Comparing `features.csv` directly with SonicLens
(30 s clips) would therefore mix two effects. The comparison uses three feature sets on the
same 11,093 tracks:

| Target | librosa, full track (FMA) | librosa, 30 s clip (recomputed) | **SonicLens, 30 s clip** |
|---|---|---|---|
| energy | 0.792 | 0.726 | **0.746** |
| danceability | 0.516 | 0.517 | **0.644** |
| valence | 0.457 | 0.338 | **0.417** |
| acousticness | 0.681 | 0.625 | **0.658** |
| instrumentalness | 0.419 | 0.371 | **0.390** |
| liveness | 0.240 | 0.102 | **0.103** |
| speechiness | 0.487 | 0.426 | **0.510** |

(Cross-validated R². Feature counts: librosa 518, SonicLens 257.)

- **The analysed span matters:** the same librosa features lose 0.069 R² on average when
  computed on the clip. The largest drops are liveness (-0.139) and valence (-0.118).
- **On identical audio, SonicLens features beat librosa on 6 of 7 targets** (by more than one
  standard error, ≈0.012), with half as many inputs. The biggest gains are danceability +0.127,
  speechiness +0.085 and valence +0.079. Liveness is a tie.
- SonicLens on 30 s clips even beats librosa on full tracks for danceability and speechiness.

### 2. Ablation by descriptor family

Change in CV R² when one family of SonicLens inputs is removed (negative = the family helps).

| Removed (n inputs) | energy | dance. | valence | acoust. | instr. | liveness | speech. |
|---|---|---|---|---|---|---|---|
| spectral (105) | -0.009 | -0.002 | +0.000 | -0.010 | -0.014 | -0.007 | -0.011 |
| timbre (85) | -0.001 | -0.006 | +0.001 | -0.024 | -0.040 | -0.004 | -0.001 |
| dynamics (8) | -0.001 | -0.004 | +0.002 | +0.001 | +0.001 | -0.002 | -0.001 |
| rhythm (37) | -0.028 | **-0.108** | **-0.081** | -0.008 | -0.012 | -0.002 | **-0.057** |
| tonal (22) | -0.004 | -0.003 | -0.029 | +0.001 | +0.001 | -0.001 | -0.015 |

- **Rhythm** descriptors matter most for danceability, valence, speechiness and energy. The
  rhythm family includes beat-synchronous loudness and band ratios, so it carries some energy
  and timbre information as well as timing.
- **Timbre** (including MFCC) matters most for instrumentalness and acousticness.
- **Tonal** descriptors help valence and speechiness.
- **Dynamics** is redundant with the rest. Families overlap, so removing one lets the others
  compensate, and changes below ≈0.012 are within fold-to-fold noise.

### 3. Permutation importance

Computed on the test set for the deployed models, top 15 per target
(`reports/importance_<target>.png`). This is a diagnostic only; nothing was selected from it.
Leading inputs:
- `onset_rate` for energy and valence;
- `tempo_confidence` and `danceability_dfa` for danceability;
- high-band spectral valleys and low-band energy for acousticness;
- high-band energy variation and MFCCs for instrumentalness;
- `tempo_confidence`, `onset_rate` and `chords_changes_rate` for speechiness.

Correlated inputs (the five statistics of one descriptor) share importance, so the ablation is
the better guide to the value of a family.

## Example: an unseen full-length song

End-to-end check through the web UI: "Bloops, Bleeps, Bongos and Brass" by Coconut Monkeyrocket
(FMA track 119050, CC BY-NC, 3:59). This artist is in the held-out test set. The full-length
track came from FMA's `fma_full.zip`, and its Echo Nest labels describe the full track.

| | SonicLens | Echo Nest |
|---|---|---|
| tempo (BPM) | 123.2 | 123.0 |
| energy | 0.69 | 0.89 |
| danceability | 0.73 | 0.84 |
| valence | 0.50 | 0.97 |
| acousticness | 0.38 | 0.12 |
| instrumentalness | 0.64 | 0.52 |
| liveness | 0.18 | 0.06 |
| speechiness | 0.11 | 0.03 |

The direction is right for most descriptors, but values are compressed towards the middle. The
very high valence is clearly underestimated, consistent with valence's moderate test R² (0.48).

## Limitations

- **Clip vs. full track.** Echo Nest labels were computed on full tracks, while the model sees a
  30 s window (the centre of the track, matching FMA's clips). Parts of a track outside the window
  cannot influence predictions. Experiment 1 shows this costs ~0.07 R² on average, and most of
  liveness's predictability.
- **Catalogue.** FMA is mostly independent, Creative Commons-licensed music. Genre and production
  coverage differs from mainstream commercial catalogues, so accuracy on such music is unknown.
- **What the model predicts.** Predictions approximate Echo Nest-style descriptors as distributed
  in FMA. They are not claimed to match the descriptors of any other service.
- **Noisy targets.** Liveness (test R² 0.12) is essentially not learnable from the clip.
  Instrumentalness has ~2,000 tracks labelled exactly 0 whose audio predicts a wide range of
  values, which suggests label noise.
- **Synthetic key and metre validation.** The key profile was chosen on synthetic audio.
  Time signature and downbeats have no ground truth in FMA.
- **Search space.** Several targets selected the most regularised edge of the hyperparameter
  space (`min_samples_leaf` 100, learning rate 0.03), so a wider search might gain a little.

## Reproducing

```bash
cd backend
.venv/bin/python -m training.download metadata   # ~1.5 min
.venv/bin/python -m training.download targets    # seconds
.venv/bin/python -m training.download audio      # ~7.5 min, 11.4 GiB
.venv/bin/python -m training.extract             # ~50 min
.venv/bin/python -m training.train               # ~53 min
.venv/bin/python -m training.librosa_clips       # ~30 min (evaluation only)
.venv/bin/python -m training.evaluate            # ~25 min
```

Times were measured on a 6-core Apple Silicon Mac with 8 GB RAM. Splits, the search and
all models are seeded; the splits are reproduced exactly.
