"""Extraction sanity checks on synthetic audio with known properties."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.core.audio import load_audio
from app.core.errors import AudioDecodeError, AudioTooShortError, SilentAudioError
from app.core.windowing import model_window
from app.extraction import MODEL_FEATURE_NAMES, extract_features
from app.extraction.features import FAMILIES, MODEL_FEATURE_FAMILIES
from app.extraction.registry import BY_KEY, keys_in_group
from tests.synth import SR, click_track, sine, write_wav

BACKEND_DIR = Path(__file__).resolve().parents[1]

# EBU Tech 3341 requires +-0.1 LU on its reference signals.
LOUDNESS_TOLERANCE_LU = 0.1


@pytest.fixture(scope="module")
def click_120():
    return extract_features(click_track(120, 4), SR)


@pytest.fixture(scope="module")
def click_120_waltz():
    return extract_features(click_track(120, 3), SR)


@pytest.fixture(scope="module")
def a4():
    return extract_features(sine(440.0, 10.0), SR)


# --- Known properties ---------------------------------------------------------------------


def test_click_track_tempo(click_120):
    assert abs(click_120["musical"]["tempo"] - 120) <= 2


def test_click_track_time_signature_and_downbeats(click_120, click_120_waltz):
    assert click_120["musical"]["time_signature"] == 4
    assert click_120_waltz["musical"]["time_signature"] == 3
    # Accented beats start at t=0, one bar every 2 s at 120 BPM in 4/4.
    downbeats = click_120["research"]["downbeats"]
    assert len(downbeats) >= 10
    assert all(abs(t / 2.0 - round(t / 2.0)) * 2.0 < 0.07 for t in downbeats)


def test_click_track_beat_grid(click_120):
    beats = np.array(click_120["research"]["beat_grid"])
    assert len(beats) >= 55
    assert np.allclose(np.diff(beats), 0.5, atol=0.03)


def test_a4_sine_chroma_peaks_at_a(a4):
    chroma = a4["research"]["chroma_vector"]
    assert len(chroma) == 12
    assert int(np.argmax(chroma)) == 9  # C=0 ... A=9
    assert chroma[9] == pytest.approx(1.0)


def test_a4_sine_brightness_and_tuning(a4):
    assert a4["signal"]["spectral_centroid"] == pytest.approx(440, abs=5)
    assert a4["signal"]["tuning_frequency"] == pytest.approx(440, abs=2)


def test_calibrated_tone_loudness():
    # EBU Tech 3341 case: stereo 1 kHz sine at -23 dBFS reads -23 LUFS.
    tone = sine(1000.0, 20.0, amplitude=10 ** (-23 / 20))
    result = extract_features(np.stack([tone, tone], axis=1), SR)
    assert result["musical"]["loudness"] == pytest.approx(-23.0, abs=LOUDNESS_TOLERANCE_LU)


def test_mono_is_treated_as_dual_mono_for_loudness():
    tone = sine(1000.0, 10.0, amplitude=10 ** (-23 / 20))
    mono = extract_features(tone, SR)["musical"]["loudness"]
    stereo = extract_features(np.stack([tone, tone], axis=1), SR)["musical"]["loudness"]
    assert mono == pytest.approx(stereo, abs=0.01)


def test_duration_from_file_is_exact(tmp_path):
    path = write_wav(tmp_path / "tone.wav", sine(440.0, 7.5))
    audio, sr = load_audio(path)
    assert extract_features(audio, sr)["musical"]["duration_ms"] == 7500


def test_other_sample_rates_and_channel_layouts():
    sr = 22050
    tone = sine(440.0, 6.0, sr=sr)
    stereo = np.stack([tone, 0.5 * tone], axis=1)
    for audio in (tone, stereo):
        result = extract_features(audio, sr)
        assert result["musical"]["duration_ms"] == 6000
        assert int(np.argmax(result["research"]["chroma_vector"])) == 9


# --- Errors ------------------------------------------------------------------------------


def test_too_short_audio_is_rejected():
    with pytest.raises(AudioTooShortError):
        extract_features(sine(440.0, 2.5), SR)


def test_silent_audio_is_rejected():
    with pytest.raises(SilentAudioError):
        extract_features(np.zeros(10 * SR, dtype=np.float32), SR)


def test_partially_silent_audio_gives_finite_output():
    audio = np.concatenate([np.zeros(5 * SR, dtype=np.float32), sine(440.0, 5.0)])
    result = extract_features(audio, SR)
    assert all(math.isfinite(v) for v in result["model_input"]["features"].values())


def test_undecodable_file_is_rejected(tmp_path):
    path = tmp_path / "not_audio.mp3"
    path.write_bytes(b"this is not audio" * 100)
    with pytest.raises(AudioDecodeError):
        load_audio(path)


# --- Model window -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "duration_s, expected_s",
    [
        (20.0, (0.0, 20.0)),  # shorter than a clip: whole track
        (30.6, (0.0, 30.6)),  # clip-length (FMA clips decode to ~30 s): whole track
        (70.0, (20.0, 50.0)),  # FMA rule: start = duration // 2 - 15
        (245.9, (107.0, 137.0)),
    ],
)
def test_model_window_follows_fma_clip_rule(duration_s, expected_s):
    start, end = model_window(int(round(duration_s * SR)), SR)
    assert (start / SR, end / SR) == pytest.approx(expected_s)


def test_long_track_uses_window_for_model_and_full_track_for_display():
    # 40 s: 20 s of A4 then 20 s of C5; the window is 5-35 s.
    audio = np.concatenate([sine(440.0, 20.0), sine(523.25, 20.0)])
    result = extract_features(audio, SR)
    assert result["model_input"]["window"] == {"start_s": 5.0, "end_s": 35.0}
    assert result["musical"]["duration_ms"] == 40000


# --- Output contract ----------------------------------------------------------------------


def _check_value(key, value):
    d = BY_KEY[key]
    if d.value_type == "string":
        assert isinstance(value, str), key
        return
    if d.value_type == "object":
        return
    values = value if d.value_type.startswith("float[") else [value]
    if d.value_type.startswith("float[") and d.value_type != "float[]":
        assert len(values) == int(d.value_type[6:-1]), key
    expected = int if d.value_type == "int" else float
    for v in values:
        assert isinstance(v, expected), key
        assert math.isfinite(v), key
        assert d.min is None or v >= d.min, f"{key}={v} < {d.min}"
        assert d.max is None or v <= d.max, f"{key}={v} > {d.max}"


@pytest.mark.parametrize("fixture", ["click_120", "a4"])
def test_output_matches_registry(fixture, request):
    result = request.getfixturevalue(fixture)
    for group in ("musical", "research", "signal"):
        assert list(result[group]) == keys_in_group(group), group
        for key, value in result[group].items():
            _check_value(key, value)
    assert result["musical"]["mode"] in ("major", "minor")
    assert len(result["research"]["mfcc"]["mean"]) == 13
    assert len(result["research"]["mfcc"]["std"]) == 13


def test_model_input_vector(click_120):
    features = click_120["model_input"]["features"]
    assert list(features) == MODEL_FEATURE_NAMES
    assert len(set(MODEL_FEATURE_NAMES)) == len(MODEL_FEATURE_NAMES)
    assert all(isinstance(v, float) and math.isfinite(v) for v in features.values())
    assert set(MODEL_FEATURE_FAMILIES.values()) == set(FAMILIES)


def test_deterministic(click_120):
    again = extract_features(click_track(120, 4), SR)
    assert json.dumps(again, sort_keys=True) == json.dumps(click_120, sort_keys=True)


def test_output_is_json_serialisable(click_120):
    json.dumps(click_120, allow_nan=False)


def test_cli_prints_json(tmp_path):
    path = write_wav(tmp_path / "tone.wav", sine(440.0, 5.0))
    out = subprocess.run(
        [sys.executable, "-m", "app.extraction", str(path)],
        cwd=BACKEND_DIR, capture_output=True, text=True, check=True,
    )
    result = json.loads(out.stdout)
    assert result["musical"]["duration_ms"] == 5000


def test_cli_reports_audio_errors(tmp_path):
    path = write_wav(tmp_path / "short.wav", sine(440.0, 1.0))
    out = subprocess.run(
        [sys.executable, "-m", "app.extraction", str(path)], cwd=BACKEND_DIR, capture_output=True, text=True
    )
    assert out.returncode == 1
    assert "at least 3 s" in out.stderr


def test_signal_families_match_model_input_families():
    from app.extraction.features import FAMILIES
    from app.extraction.registry import DESCRIPTORS

    # Display keys that correspond to a model input (frame series summarised with .mean, or a scalar).
    aliases = {"rms_mean": "rms.mean", "tempo_stability": "bpm_first_peak_weight",
               "energy_band_low": "energy_band_ratio_low.mean"}
    for d in DESCRIPTORS:
        assert (d.family is not None) == (d.group == "signal"), d.key
        if d.group != "signal":
            continue
        assert d.family in FAMILIES
        name = aliases.get(d.key, d.key)
        for candidate in (name, f"{name}.mean"):
            if candidate in MODEL_FEATURE_FAMILIES:
                assert MODEL_FEATURE_FAMILIES[candidate] == d.family, d.key
