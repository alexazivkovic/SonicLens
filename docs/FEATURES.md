# Descriptors

SonicLens returns four groups of descriptors. This page documents every one of them. The
machine-readable version is `backend/app/extraction/registry.py`, served at `GET /features`.

| Group | What | Computed by | Computed on |
|---|---|---|---|
| **musical** | Core musical descriptors (tempo, key, loudness, ...) | DSP (Essentia) | full track |
| **perceptual** | Energy, danceability, valence, ... | learned model | 30 s model window |
| **research** | Chroma, beat grid, downbeats, MFCC | DSP (Essentia) | full track |
| **signal** | Spectral, timbre, dynamics, rhythm and tonal statistics | DSP (Essentia) | full track |

Column legend for the tables below:
- **Model input**: whether the descriptor (or its frame series) is part of the model input
  vector. Model inputs are computed on the 30 s model window, not the full track (see
  `ARCHITECTURE.md`).
- **Tier**: `core` descriptors are shown prominently in the UI. `advanced` ones sit in the
  collapsible table.
- **Estimated**: heuristic values that are less reliable than the rest. The UI marks them.

## Analysis parameters

All audio is converted to 44.1 kHz. Descriptors are computed on the mono downmix, except EBU R128
loudness, which uses the stereo signal (mono sources are treated as dual-mono).

| Analysis | Frame size | Hop | Window |
|---|---|---|---|
| Spectral, timbre, MFCC, RMS | 2048 | 1024 | Hann |
| Tonal (HPCP, chords, tuning) | 4096 | 2048 | Blackman-Harris 62 dB |
| Rhythm, key, loudness, onsets, DFA | Essentia algorithm defaults | | |

Frame-level values are shown as their mean over the track. In the model input vector they are
summarised with mean, std, median, 10th and 90th percentile.

## Musical descriptors (DSP, full track)

| Key | Definition | Why it is included | Unit / range | Algorithm | Model input | Est. | Tier |
|---|---|---|---|---|---|---|---|
| `duration_ms` | Length of the audio (sample count / sample rate) | Basic context for every other value | ms, ≥ 0 | - | no | no | core |
| `tempo` | Tempo of the beat | The most-asked-for musical property; drives danceability and energy | BPM, 0-250 | `RhythmExtractor2013` (multifeature) | yes | no | core |
| `key` | Tonic, spelled with sharps (C, C#, ... B) | Essential for DJs, musicians and harmonic analysis | pitch class | `KeyExtractor` (edma) | no | no | core |
| `mode` | Major or minor | Mode is a classic correlate of perceived valence | major / minor | `KeyExtractor` (edma) | yes (`mode_major`) | no | core |
| `key_confidence` | Strength of the key estimate | Says how tonal and unambiguous the music is; low for atonal or noisy music | 0-1 | `KeyExtractor` (edma) | yes (`key_strength`) | no | core |
| `loudness` | Integrated programme loudness (EBU R128 / ITU-R BS.1770) | The broadcast standard for loudness; relates to energy and mastering style | LUFS, -70 to 10 | `LoudnessEBUR128` | yes (`loudness_integrated`) | no | core |
| `time_signature` | Beats per bar | Metre is part of a track's rhythmic identity | 3, 4, 5 or 7 | `BeatsLoudness` + `Beatogram` + `Meter` | yes | **yes** | core |

## Perceptual descriptors (model predictions)

All are predicted from the model input vector by one `HistGradientBoostingRegressor` per
descriptor, and clipped to 0-1. Test R² is measured on a held-out set of artists. It says how far
each value can be trusted (`GET /model`, `backend/reports/SUMMARY.md`).

| Key | Definition | Why it is included | Test R² |
|---|---|---|---|
| `energy` | Perceived intensity and activity: fast, loud, dense and noisy music scores high | A primary axis for playlisting and mood | 0.76 |
| `danceability` | Suitability for dancing: tempo, beat strength, rhythmic regularity | Widely used for recommendation and DJ research | 0.65 |
| `valence` | Musical positiveness: cheerful/euphoric (high) vs. sad/tense/angry (low) | The standard valence axis of music-emotion research | 0.48 |
| `acousticness` | Confidence that the recording is acoustic (no electronic instruments or heavy production) | Separates acoustic from produced/electronic material | 0.62 |
| `instrumentalness` | Likelihood that the track has no vocals (> 0.5 suggests instrumental) | Vocal presence matters for background music and analysis | 0.44 |
| `liveness` | Likelihood of a live recording with an audience (> 0.8 strongly suggests live) | Distinguishes studio from live recordings | 0.12 (low) |
| `speechiness` | Presence of spoken words (> 0.66 mostly speech; 0.33-0.66 music and speech, e.g. rap) | Detects spoken word, podcasts and rap | 0.60 |

Liveness is barely learnable from a 30 s centre window: live cues (applause, crowd noise) sit
mostly at the start and end of a track. See `TRAINING.md`.

## Research descriptors (DSP, full track)

| Key | Definition | Why it is included | Unit / range | Algorithm | Model input | Est. | Tier |
|---|---|---|---|---|---|---|---|
| `chroma_vector` | Mean harmonic pitch-class profile, 12 bins C-B, strongest bin = 1 | Shows the tonal centre and harmonic content at a glance | 12 values, 0-1 | `HPCP` (12 bins, tuned to A4 = 440 Hz) | yes (`chroma_C` ... `chroma_B`) | no | core |
| `beat_grid` | Beat positions | Needed for beat-synchronous analysis, DJ tools and visualisation | seconds | `RhythmExtractor2013` (multifeature) | no (summarised by tempo and beat statistics) | no | core |
| `downbeats` | First beat of each bar | Bar-level structure for alignment and visualisation | seconds | `BeatsLoudness` + `Meter` (heuristic, see below) | no | **yes** | core |
| `mfcc` | Mel-frequency cepstral coefficients 0-12, mean and std over frames | The standard compact timbre representation in MIR research | 13 + 13 values | `MFCC` | yes (all 5 statistics of each coefficient) | no | core |

## Signal descriptors (DSP, full track)

These were chosen because they are known to correlate with the perceptual descriptors.
Descriptors marked `core` are highlighted in the UI under "Sound character".

**Spectral**

| Key | Definition | Why it is included | Unit / range | Algorithm | Tier |
|---|---|---|---|---|---|
| `spectral_centroid` | "Brightness": the spectrum's centre of mass | Bright vs. dark timbre; correlates with energy and acousticness | Hz, 0-22050 | `Centroid` | **core** |
| `spectral_rolloff` | Frequency below which 85% of spectral energy lies | Bandwidth of the sound; high for noisy/percussive material | Hz, 0-22050 | `RollOff` | advanced |
| `spectral_flux` | Frame-to-frame change of the spectrum (L2) | Activity and change; correlates with energy | ≥ 0 | `Flux` | advanced |
| `spectral_flatness` | How noise-like (flat) vs. tonal (peaky) the spectrum is | Noise vs. tone; distorted/electronic vs. acoustic | 0-1 | `FlatnessDB` | advanced |
| `spectral_complexity` | Number of spectral peaks | Density of the texture (many instruments vs. few) | peaks, ≥ 0 | `SpectralComplexity` | advanced |
| `spectral_contrast` | Peak-to-valley contrast in 6 octave bands | Separates harmonic from noisy content per band; strong for acousticness | 6 values | `SpectralContrast` | advanced |
| `spectral_valleys` | Valley level in the same 6 bands | Noise floor per band; complements contrast | 6 values | `SpectralContrast` | advanced |
| `energy_band_low` | Share of energy in 20-150 Hz | Bass/kick weight; production style and danceability | 0-1 | `EnergyBandRatio` | advanced |
| `energy_band_mid_low` | Share of energy in 150-800 Hz | Body of instruments and voice | 0-1 | `EnergyBandRatio` | advanced |
| `energy_band_mid_high` | Share of energy in 0.8-4 kHz | Presence region; voice intelligibility | 0-1 | `EnergyBandRatio` | advanced |
| `energy_band_high` | Share of energy in 4-20 kHz | Cymbals, sibilance, "air"; brightness | 0-1 | `EnergyBandRatio` | advanced |

**Timbre and noise**

| Key | Definition | Why it is included | Unit / range | Algorithm | Tier |
|---|---|---|---|---|---|
| `zero_crossing_rate` | Rate of waveform sign changes | Noisiness and percussiveness; classic speech/music cue | 0-1 | `ZeroCrossingRate` | advanced |
| `dissonance` | Sensory roughness of simultaneous spectral peaks | Harshness vs. consonance; relates to valence and energy | 0-1 | `SpectralPeaks` + `Dissonance` | **core** |
| `pitch_salience` | How clearly a pitch is perceived | Harmonic (pitched) vs. inharmonic sound; relates to instrumentalness and speechiness | 0-1 | `PitchSalience` | advanced |
| `hfc` | High-frequency content, weighted towards high bins | Percussive attacks; energy | ≥ 0 | `HFC` | advanced |

**Dynamics**

| Key | Definition | Why it is included | Unit / range | Algorithm | Tier |
|---|---|---|---|---|---|
| `rms_mean` | Mean frame RMS amplitude | Overall level | 0-1 | `RMS` | advanced |
| `rms_std` | Standard deviation of frame RMS | Level variation (steady vs. dynamic) | 0-1 | `RMS` | advanced |
| `loudness_range` | EBU R128 loudness range (LRA) | "Dynamic range": compressed pop vs. dynamic classical/live | LU, ≥ 0 | `LoudnessEBUR128` | **core** |
| `dynamic_complexity` | Average deviation from the global loudness | Amount of loudness fluctuation | dB, ≥ 0 | `DynamicComplexity` | advanced |

**Rhythm**

| Key | Definition | Why it is included | Unit / range | Algorithm | Tier |
|---|---|---|---|---|---|
| `onset_rate` | "Note density": onsets per second | Busy vs. sparse music; the single strongest input for energy and valence | onsets/s, ≥ 0 | `OnsetRate` | **core** |
| `beats_loudness` | Mean spectral energy at the beats | Beat strength; danceability | ≥ 0 | `BeatsLoudness` | advanced |
| `tempo_stability` | Weight of the main peak of the beat-interval histogram | Steady (electronic, pop) vs. rubato/free tempo; danceability | 0-1 | `BpmHistogramDescriptors` | **core** |
| `bpm_second_peak_weight` | Weight of the second histogram peak | Tempo ambiguity (e.g. half-time feel) | 0-1 | `BpmHistogramDescriptors` | advanced |
| `danceability_dfa` | Essentia's detrended-fluctuation danceability (roughly 0-3). A DSP input, not the learned danceability | Rhythmic regularity at many time scales; a strong danceability input | ≥ 0 | `Danceability` | advanced |

**Tonal**

| Key | Definition | Why it is included | Unit / range | Algorithm | Tier |
|---|---|---|---|---|---|
| `hpcp_entropy` | Entropy of the pitch-class profile per frame | Harmonic clarity vs. diffuseness/noise | bits, 0-3.585 | `HPCP` + `Entropy` | advanced |
| `chords_changes_rate` | Share of analysis windows where the chord changes | Harmonic rhythm; also separates speech from music | 0-1 | `ChordsDetection` + `ChordsDescriptors` | advanced |
| `chords_number_rate` | Distinct frequent chords relative to chord estimates | Harmonic variety | 0-1 | `ChordsDetection` + `ChordsDescriptors` | advanced |
| `tuning_frequency` | Estimated reference frequency of A4 | Detects non-standard tuning; context for key and chroma | Hz, 400-480 | `TuningFrequency` | advanced |

All signal descriptors are model inputs (as frame statistics or track-level values).

## Key: choice of profile

`KeyExtractor` supports several key profiles. The default is **`edma`**, chosen by
`backend/training/eval_key_profiles.py` (results in `backend/reports/key_profiles.json`).
FMA has no key annotations, so the evaluation uses 864 synthesised pieces covering all 24 keys.
Each piece is a looped chord progression with a melody, rendered with 3 timbres, with and
without +30 cent detuning, and with and without noise percussion. It is scored with the MIREX
weighted key score (correct 1.0, fifth 0.5, relative 0.3, parallel 0.2):

| Profile | Accuracy | MIREX weighted |
|---|---|---|
| **edma** | **0.82** | **0.86** |
| krumhansl | 0.72 | 0.80 |
| temperley | 0.61 | 0.71 |

All three profiles are perfect on classical cadences (I-IV-V-I, i-iv-V-i). They differ on Aeolian
minor progressions without a leading tone (i-VI-III-VII, i-VII-VI-VII), which are common in
popular and electronic music. There, `edma` (designed for electronic dance music) is clearly the
most robust. Limitation: synthetic audio is much cleaner than real recordings, so absolute
accuracies are optimistic; only the ranking is used.

## Estimated descriptors

- **`time_signature`**: Essentia's `Meter` algorithm, applied to a `Beatogram` built from
  `BeatsLoudness` at the beat positions of `RhythmExtractor2013`. On synthetic accented click
  tracks it correctly identifies 3/4 and 4/4 at 90-140 BPM. On real music, however, `Meter` often
  returns a multiple or divisor of the bar length: on 80 random FMA clips it returned 2 (31x),
  4 (23x), 8 (8x), 3 (8x), 6 (5x), 12 (2x) and 5, 7, 10 once each. The raw value is therefore
  reduced to the conventional meter classes: multiples of 3 -> **3**, multiples of 5 -> **5**,
  7 -> **7**, and anything else (2, 4, 8, ...) -> **4**. With fewer than 16 beats, or a raw value
  outside 2-12, the result falls back to 4. Over the 13,103 training clips this gives 4: 83%,
  3: 13%, 5: 3%, 7: 1%. FMA has no time-signature annotations, so accuracy on real music is
  unvalidated; treat it as a rough estimate.
- **`downbeats`**: every n-th beat, where n is the estimated time signature. The phase (which
  beat is "one") is the one whose beats carry the most low-frequency energy (20-150 Hz band of
  `BeatsLoudness`), since kick drums and bass notes tend to land on the downbeat.

## Validation

- **Tempo**: agrees with the Echo Nest tempo within ±4% on 56.4% of the 13,103 training clips,
  and on 69.5% counting double/half tempo. The remaining disagreements are mostly 3:2 / 2:3
  confusions, plus a few quantised high-tempo estimates (`backend/reports/tempo_agreement.png`).
- **Loudness**: the EBU Tech 3341 reference tones (stereo 1 kHz sine at -23, -33 and -18 dBFS)
  read -22.99, -32.99 and -17.99 LUFS. The test suite enforces **±0.1 LU**, the tolerance EBU
  Tech 3341 itself requires.
- **Synthetic tests** (`backend/tests/test_extraction.py`):
  - a 120 BPM click track gives a tempo within ±2 BPM, a 0.5 s beat grid, a 4/4 metre and
    downbeats on the accented beats; a 3/4 click track gives a 3/4 metre;
  - an A4 sine peaks at chroma bin A, with a 440 Hz centroid and tuning;
  - `duration_ms` of a file is exact, and repeated runs give identical output.

## Model input vector

The model input is a flat, ordered, named vector of **257 values**, computed on the model window.
The ordered names are `MODEL_FEATURE_NAMES` in `backend/app/extraction/features.py`. They are
saved with every model artifact and validated when the model loads.

- **Frame-level series** (and beat-level ones), each as mean, std, median, p10 and p90
  (`<name>.<stat>`): spectral centroid, rolloff, flux, flatness, complexity, contrast and valleys
  (6 bands each), energy band ratios (4), zero-crossing rate, dissonance, pitch salience, HFC,
  MFCC 0-12, RMS, beat loudness and its 5 band ratios, HPCP entropy.
- **Track-level values**: tempo, tempo confidence, BPM histogram first/second peak weight, onset
  rate, DFA danceability, time signature, integrated loudness, loudness range, dynamic
  complexity, key strength, mode (major = 1), tuning frequency, chord change rate, chord number
  rate.
- **Chroma**: the 12 mean HPCP bins (C-B).

Every input belongs to one of five families (spectral, timbre, dynamics, rhythm, tonal). The
ablation experiments remove one family at a time.

Deliberately **not** model inputs:
- `duration_ms`: the window is ~30 s for almost every input, so it carries no information.
- The key root (pitch class): it is a circular category, not a quantity.
- `beat_grid` and `downbeats`: variable-length lists; their information is summarised by tempo,
  tempo stability and beat loudness.

## Robustness

- Any sample rate, mono or stereo, is accepted (resampled to 44.1 kHz).
- Audio shorter than 3 s is rejected (`AudioTooShortError`, HTTP 422). The training scripts skip
  such clips with a log line.
- Audio with a peak level below -80 dBFS is rejected as silent (`SilentAudioError`, HTTP 422).
- Partially silent audio is analysed normally. Any non-finite intermediate value is replaced
  with 0.
- Integrated loudness below the EBU R128 absolute gate is reported as -70 LUFS.
- Extraction is deterministic, and `EXTRACTOR_VERSION` is bumped whenever its output can change.
