"""The single extraction entry point shared by the training scripts and the API.

`extract_features(audio, sr)` returns:

- `musical`, `research`, `signal`: displayed descriptors, computed on the FULL track.
- `model_input`: the flat, ordered, named model input vector, computed on the model window
  (the centred 30 s that matches FMA's training clips; see app.core.windowing).
"""

import numpy as np

from app.core.audio import SAMPLE_RATE, prepare
from app.core.version import ESSENTIA_VERSION, EXTRACTOR_VERSION
from app.core.windowing import model_window
from app.extraction.analysis import ENERGY_BANDS, N_BEAT_BANDS, N_CONTRAST_BANDS, N_MFCC, PITCH_CLASSES, analyse

STATS = ("mean", "std", "median", "p10", "p90")

# Every model input belongs to one descriptor family; evaluate.py ablates one family at a time.
FAMILIES = ("spectral", "timbre", "dynamics", "rhythm", "tonal")

# Frame-level (or beat-level) series, summarised with STATS. Order is part of the model contract.
SERIES_FAMILIES: dict[str, str] = {
    "spectral_centroid": "spectral",
    "spectral_rolloff": "spectral",
    "spectral_flux": "spectral",
    "spectral_flatness": "spectral",
    "spectral_complexity": "spectral",
    **{f"spectral_contrast_{i}": "spectral" for i in range(N_CONTRAST_BANDS)},
    **{f"spectral_valley_{i}": "spectral" for i in range(N_CONTRAST_BANDS)},
    **{f"energy_band_ratio_{name}": "spectral" for name in ENERGY_BANDS},
    "zero_crossing_rate": "timbre",
    "dissonance": "timbre",
    "pitch_salience": "timbre",
    "hfc": "timbre",
    **{f"mfcc_{i:02d}": "timbre" for i in range(N_MFCC)},
    "rms": "dynamics",
    "beats_loudness": "rhythm",
    **{f"beats_loudness_band_ratio_{i}": "rhythm" for i in range(N_BEAT_BANDS)},
    "hpcp_entropy": "tonal",
}

# Track-level values. Duration and the key root are deliberately not model inputs: the window
# duration is ~30 s for almost every input, and a pitch class has no ordinal meaning.
SCALAR_FAMILIES: dict[str, str] = {
    "tempo": "rhythm",
    "tempo_confidence": "rhythm",
    "bpm_first_peak_weight": "rhythm",
    "bpm_second_peak_weight": "rhythm",
    "onset_rate": "rhythm",
    "danceability_dfa": "rhythm",
    "time_signature": "rhythm",
    "loudness_integrated": "dynamics",
    "loudness_range": "dynamics",
    "dynamic_complexity": "dynamics",
    "key_strength": "tonal",
    "mode_major": "tonal",
    "tuning_frequency": "tonal",
    "chords_changes_rate": "tonal",
    "chords_number_rate": "tonal",
}

CHROMA_NAMES = [f"chroma_{pc}" for pc in PITCH_CLASSES]

MODEL_FEATURE_FAMILIES: dict[str, str] = {
    **{f"{name}.{stat}": family for name, family in SERIES_FAMILIES.items() for stat in STATS},
    **SCALAR_FAMILIES,
    **{name: "tonal" for name in CHROMA_NAMES},
}
MODEL_FEATURE_NAMES: list[str] = list(MODEL_FEATURE_FAMILIES)


def _summarise(x: np.ndarray) -> list[float]:
    if x.size == 0:
        return [0.0] * len(STATS)
    p10, median, p90 = np.percentile(x, [10, 50, 90])
    return [float(np.mean(x)), float(np.std(x)), float(median), float(p10), float(p90)]


def model_input_vector(a: dict) -> dict[str, float]:
    """Flatten one analysis into the ordered model input vector (name -> value)."""
    values: dict[str, float] = {}
    for name in SERIES_FAMILIES:
        for stat, value in zip(STATS, _summarise(a["series"][name])):
            values[f"{name}.{stat}"] = value
    for name in SCALAR_FAMILIES:
        values[name] = a["scalars"][name]
    for name, value in zip(CHROMA_NAMES, a["chroma"]):
        values[name] = float(value)
    assert list(values) == MODEL_FEATURE_NAMES
    return values


def _mean(a: dict, name: str) -> float:
    x = a["series"][name]
    return float(np.mean(x)) if x.size else 0.0


def _display(a: dict, duration_ms: int) -> tuple[dict, dict, dict]:
    s = a["scalars"]
    musical = {
        "duration_ms": duration_ms,
        "tempo": s["tempo"],
        "key": a["key"],
        "mode": a["mode"],
        "key_confidence": s["key_strength"],
        "loudness": s["loudness_integrated"],
        "time_signature": int(s["time_signature"]),
    }
    research = {
        "chroma_vector": [float(v) for v in a["chroma"]],
        "beat_grid": [float(t) for t in a["beats"]],
        "downbeats": [float(t) for t in a["downbeats"]],
        "mfcc": {
            "mean": [_mean(a, f"mfcc_{i:02d}") for i in range(N_MFCC)],
            "std": [float(np.std(a["series"][f"mfcc_{i:02d}"])) for i in range(N_MFCC)],
        },
    }
    signal = {
        "spectral_centroid": _mean(a, "spectral_centroid"),
        "spectral_rolloff": _mean(a, "spectral_rolloff"),
        "spectral_flux": _mean(a, "spectral_flux"),
        "spectral_flatness": _mean(a, "spectral_flatness"),
        "spectral_complexity": _mean(a, "spectral_complexity"),
        "spectral_contrast": [_mean(a, f"spectral_contrast_{i}") for i in range(N_CONTRAST_BANDS)],
        "spectral_valleys": [_mean(a, f"spectral_valley_{i}") for i in range(N_CONTRAST_BANDS)],
        **{f"energy_band_{name}": _mean(a, f"energy_band_ratio_{name}") for name in ENERGY_BANDS},
        "zero_crossing_rate": _mean(a, "zero_crossing_rate"),
        "dissonance": _mean(a, "dissonance"),
        "pitch_salience": _mean(a, "pitch_salience"),
        "hfc": _mean(a, "hfc"),
        "rms_mean": _mean(a, "rms"),
        "rms_std": float(np.std(a["series"]["rms"])),
        "loudness_range": s["loudness_range"],
        "dynamic_complexity": s["dynamic_complexity"],
        "onset_rate": s["onset_rate"],
        "beats_loudness": _mean(a, "beats_loudness"),
        "tempo_stability": s["bpm_first_peak_weight"],
        "bpm_second_peak_weight": s["bpm_second_peak_weight"],
        "danceability_dfa": s["danceability_dfa"],
        "hpcp_entropy": _mean(a, "hpcp_entropy"),
        "chords_changes_rate": s["chords_changes_rate"],
        "chords_number_rate": s["chords_number_rate"],
        "tuning_frequency": s["tuning_frequency"],
    }
    return musical, research, signal


def extract_features(audio: np.ndarray, sr: int) -> dict:
    """Extract all descriptors from raw audio.

    `audio` is 1-D (mono) or shaped (n_samples, n_channels), at any sample rate. Raises
    app.core.errors.AudioError subclasses for audio that is too short, silent or malformed.
    Deterministic: the same input always gives the same output.
    """
    n_samples = np.asarray(audio).shape[0]
    duration_ms = int(round(n_samples / sr * 1000))
    mono, stereo = prepare(audio, sr)

    full = analyse(mono, stereo)
    start, end = model_window(len(mono), SAMPLE_RATE)
    window = full if (start, end) == (0, len(mono)) else analyse(mono[start:end], stereo[start:end])

    musical, research, signal = _display(full, duration_ms)
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "essentia_version": ESSENTIA_VERSION,
        "musical": musical,
        "research": research,
        "signal": signal,
        "model_input": {
            "window": {"start_s": start / SAMPLE_RATE, "end_s": end / SAMPLE_RATE},
            "features": model_input_vector(window),
        },
    }
