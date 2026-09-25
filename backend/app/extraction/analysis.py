"""Raw descriptor computation with Essentia.

`analyse()` runs every algorithm once over a signal and returns frame-level series, scalar
values and a few structured results. It knows nothing about how results are displayed or fed
to the model; `app.extraction.features` builds both views from its output.
"""

import essentia
import essentia.standard as es
import numpy as np

from app.core.audio import SAMPLE_RATE as SR

essentia.log.infoActive = False

FRAME_SIZE = 2048
HOP_SIZE = 1024
TONAL_FRAME_SIZE = 4096
TONAL_HOP_SIZE = 2048

KEY_PROFILE = "edma"  # chosen with training/eval_key_profiles.py, see docs/FEATURES.md
N_MFCC = 13
N_CONTRAST_BANDS = 6
ENERGY_BANDS = {
    "low": (20.0, 150.0),
    "mid_low": (150.0, 800.0),
    "mid_high": (800.0, 4000.0),
    "high": (4000.0, 20000.0),
}
# BeatsLoudness default frequency bands (Hz): 20-150-400-3200-7000-22000.
N_BEAT_BANDS = 5
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
# Essentia's HPCP bin 0 is the reference frequency (A4 = 440 Hz); C is three bins above.
_HPCP_C_OFFSET = 3
# Beatogram/Meter need a few bars of beats to say anything about metre.
_MIN_BEATS_FOR_METER = 16
_DEFAULT_TIME_SIGNATURE = 4


def _finite(x):
    """Replace NaN and +-inf (e.g. from degenerate silent frames) with 0."""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _spectral_frames(mono: np.ndarray) -> dict[str, np.ndarray]:
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum(size=FRAME_SIZE)
    centroid = es.Centroid(range=SR / 2)
    rolloff = es.RollOff(sampleRate=SR)
    flux = es.Flux()
    flatness = es.FlatnessDB()
    complexity = es.SpectralComplexity(sampleRate=SR)
    contrast = es.SpectralContrast(frameSize=FRAME_SIZE, sampleRate=SR)
    band_ratios = {
        name: es.EnergyBandRatio(sampleRate=SR, startFrequency=lo, stopFrequency=hi)
        for name, (lo, hi) in ENERGY_BANDS.items()
    }
    zcr = es.ZeroCrossingRate()
    rms = es.RMS()
    peaks = es.SpectralPeaks(sampleRate=SR, orderBy="frequency", minFrequency=20, maxFrequency=5000, maxPeaks=100)
    dissonance = es.Dissonance()
    pitch_salience = es.PitchSalience(sampleRate=SR)
    hfc = es.HFC(sampleRate=SR)
    mfcc = es.MFCC(inputSize=FRAME_SIZE // 2 + 1, sampleRate=SR, numberCoefficients=N_MFCC)

    rows: dict[str, list] = {}

    def add(name: str, value) -> None:
        rows.setdefault(name, []).append(float(value))

    for frame in es.FrameGenerator(mono, frameSize=FRAME_SIZE, hopSize=HOP_SIZE, startFromZero=True):
        spec = spectrum(window(frame))
        add("spectral_centroid", centroid(spec))
        add("spectral_rolloff", rolloff(spec))
        add("spectral_flux", flux(spec))
        add("spectral_flatness", flatness(spec))
        add("spectral_complexity", complexity(spec))
        contrast_values, valley_values = contrast(spec)
        for i in range(N_CONTRAST_BANDS):
            add(f"spectral_contrast_{i}", contrast_values[i])
            add(f"spectral_valley_{i}", valley_values[i])
        for name, algo in band_ratios.items():
            add(f"energy_band_ratio_{name}", algo(spec))
        add("zero_crossing_rate", zcr(frame))
        add("rms", rms(frame))
        freqs, mags = peaks(spec)
        add("dissonance", dissonance(freqs, mags) if len(freqs) > 1 else 0.0)
        add("pitch_salience", pitch_salience(spec))
        add("hfc", hfc(spec))
        _, coeffs = mfcc(spec)
        for i in range(N_MFCC):
            add(f"mfcc_{i:02d}", coeffs[i])

    return {name: _finite(np.asarray(v)) for name, v in rows.items()}


def _tonal(mono: np.ndarray, key: str, scale: str) -> dict:
    window = es.Windowing(type="blackmanharris62")
    spectrum = es.Spectrum(size=TONAL_FRAME_SIZE)
    peaks = es.SpectralPeaks(
        sampleRate=SR, orderBy="magnitude", magnitudeThreshold=1e-5,
        minFrequency=20, maxFrequency=3500, maxPeaks=60,
    )
    whitening = es.SpectralWhitening(sampleRate=SR, maxFrequency=3500)
    hpcp = es.HPCP(
        size=12, referenceFrequency=440, harmonics=8, bandPreset=True, minFrequency=20,
        maxFrequency=3500, weightType="cosine", nonLinear=False, windowSize=1.0, normalized="unitMax",
    )
    tuning = es.TuningFrequency()
    entropy = es.Entropy()

    hpcps, entropies = [], []
    tuning_frequency = 440.0
    for frame in es.FrameGenerator(mono, frameSize=TONAL_FRAME_SIZE, hopSize=TONAL_HOP_SIZE, startFromZero=True):
        spec = spectrum(window(frame))
        freqs, mags = peaks(spec)
        if len(freqs) > 0:
            tuning_frequency, _ = tuning(freqs, mags)
        mags = whitening(spec, freqs, mags)
        h = hpcp(freqs, mags)
        hpcps.append(h)
        entropies.append(float(entropy(h / h.sum())) if h.sum() > 0 else 0.0)

    hpcp_frames = np.asarray(hpcps, dtype=np.float32)
    chroma = hpcp_frames.mean(axis=0)
    chroma = np.roll(chroma, -_HPCP_C_OFFSET)
    if chroma.max() > 0:
        chroma = chroma / chroma.max()

    chords, _ = es.ChordsDetection(hopSize=TONAL_HOP_SIZE, sampleRate=SR, windowSize=2)(hpcp_frames)
    _, number_rate, changes_rate, _, _ = es.ChordsDescriptors()(chords, key, scale)

    return {
        "hpcp_entropy": np.asarray(entropies),
        "chroma": chroma.astype(float),
        "tuning_frequency": float(tuning_frequency),
        "chords_number_rate": float(number_rate),
        "chords_changes_rate": float(changes_rate),
    }


def _meter_and_downbeats(beats: np.ndarray, beat_loudness: np.ndarray, band_ratios: np.ndarray):
    """Time signature from Essentia's Meter, downbeats from low-band beat loudness.

    Both are estimates. On real music Meter often reports a multiple or divisor of the bar
    length (2, 6, 8, 12 beats), so its result is reduced to the conventional meter classes:
    multiples of 3 -> 3, multiples of 5 -> 5, 7 -> 7, anything else (2, 4, 8, ...) -> 4. With
    too few beats, or a Meter result outside 2-12, the time signature falls back to 4.
    Downbeats are every n-th beat, where n is the time signature, starting at the phase whose
    beats carry the most low-frequency (kick/bass, 20-150 Hz) energy.
    """
    if len(beats) < _MIN_BEATS_FOR_METER:
        return _DEFAULT_TIME_SIGNATURE, np.array([])
    try:
        beatogram = es.Beatogram()(beat_loudness, band_ratios)
        meter = int(round(es.Meter()(beatogram)))
    except (RuntimeError, ValueError):
        meter = _DEFAULT_TIME_SIGNATURE
    if not 2 <= meter <= 12:
        time_signature = _DEFAULT_TIME_SIGNATURE
    elif meter % 3 == 0:
        time_signature = 3
    elif meter % 5 == 0:
        time_signature = 5
    elif meter == 7:
        time_signature = 7
    else:
        time_signature = _DEFAULT_TIME_SIGNATURE

    low_energy = beat_loudness * band_ratios[:, 0]
    phase_strength = [low_energy[p::time_signature].mean() for p in range(time_signature)]
    phase = int(np.argmax(phase_strength))
    return time_signature, beats[phase::time_signature]


def _rhythm(mono: np.ndarray) -> dict:
    bpm, beats, confidence, _, intervals = es.RhythmExtractor2013(method="multifeature")(mono)
    beats = np.asarray(beats, dtype=float)

    if len(intervals) > 0:
        _, first_weight, _, _, second_weight, _, _ = es.BpmHistogramDescriptors()(intervals)
    else:
        first_weight = second_weight = 0.0

    if len(beats) > 0:
        beat_loudness, band_ratios = es.BeatsLoudness(beats=beats.astype(np.float32).tolist(), sampleRate=SR)(mono)
        beat_loudness = np.asarray(beat_loudness, dtype=float)
        band_ratios = np.asarray(band_ratios, dtype=float).reshape(-1, N_BEAT_BANDS)
    else:
        beat_loudness = np.array([])
        band_ratios = np.zeros((0, N_BEAT_BANDS))

    time_signature, downbeats = _meter_and_downbeats(beats, beat_loudness, band_ratios)
    _, onset_rate = es.OnsetRate()(mono)
    danceability, _ = es.Danceability(sampleRate=SR)(mono)

    series = {"beats_loudness": _finite(beat_loudness)}
    for i in range(N_BEAT_BANDS):
        series[f"beats_loudness_band_ratio_{i}"] = _finite(band_ratios[:, i])

    return {
        "series": series,
        "tempo": float(bpm),
        "tempo_confidence": float(confidence),
        "bpm_first_peak_weight": float(first_weight),
        "bpm_second_peak_weight": float(second_weight),
        "onset_rate": float(onset_rate),
        "danceability_dfa": float(danceability),
        "time_signature": time_signature,
        "beats": beats,
        "downbeats": np.asarray(downbeats, dtype=float),
    }


def analyse(mono: np.ndarray, stereo: np.ndarray) -> dict:
    """Run all descriptor algorithms over one signal (44.1 kHz; mono and its stereo pair)."""
    key, scale, key_strength = es.KeyExtractor(profileType=KEY_PROFILE, sampleRate=SR)(mono)
    _, _, loudness_integrated, loudness_range = es.LoudnessEBUR128(sampleRate=SR)(stereo)
    dynamic_complexity, _ = es.DynamicComplexity(sampleRate=SR)(mono)
    tonal = _tonal(mono, key, scale)
    rhythm = _rhythm(mono)

    series = _spectral_frames(mono)
    series["hpcp_entropy"] = _finite(tonal["hpcp_entropy"])
    series.update(rhythm["series"])

    scalars = {
        "tempo": rhythm["tempo"],
        "tempo_confidence": rhythm["tempo_confidence"],
        "bpm_first_peak_weight": rhythm["bpm_first_peak_weight"],
        "bpm_second_peak_weight": rhythm["bpm_second_peak_weight"],
        "onset_rate": rhythm["onset_rate"],
        "danceability_dfa": rhythm["danceability_dfa"],
        "time_signature": float(rhythm["time_signature"]),
        "loudness_integrated": float(loudness_integrated),
        "loudness_range": float(loudness_range),
        "dynamic_complexity": float(dynamic_complexity),
        "key_strength": float(key_strength),
        "mode_major": 1.0 if scale == "major" else 0.0,
        "tuning_frequency": tonal["tuning_frequency"],
        "chords_changes_rate": tonal["chords_changes_rate"],
        "chords_number_rate": tonal["chords_number_rate"],
    }
    scalars = {k: float(_finite(v)) for k, v in scalars.items()}

    return {
        "series": series,
        "scalars": scalars,
        "key": _FLAT_TO_SHARP.get(key, key),
        "mode": scale,
        "chroma": tonal["chroma"],
        "beats": rhythm["beats"],
        "downbeats": rhythm["downbeats"],
    }
