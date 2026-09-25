"""Metadata for every displayed descriptor. Served by GET /features and checked by tests.

Groups: "musical" (core musical descriptors), "perceptual" (learned by the model),
"research" (DSP, visualised), "signal" (DSP, advanced table).
"""

from dataclasses import asdict, dataclass

GROUPS = ("musical", "perceptual", "research", "signal")


@dataclass(frozen=True)
class Descriptor:
    key: str
    group: str
    label: str
    description: str
    unit: str | None
    # Inclusive bounds of valid values; None means unbounded on that side.
    min: float | None
    max: float | None
    method: str  # "dsp" | "model"
    algorithm: str  # Essentia algorithm(s), or the model
    tier: str = "core"  # "core" | "advanced"
    estimated: bool = False
    model_input: bool = True  # whether this descriptor (or its frame series) feeds the model
    value_type: str = "float"  # "float" | "int" | "string" | "float[N]" | "float[]" | "object"
    # Signal descriptors only: the descriptor family (as used in the ablation experiments).
    family: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _musical(key, label, description, unit, lo, hi, algorithm, **kw):
    return Descriptor(key, "musical", label, description, unit, lo, hi, "dsp", algorithm, **kw)


def _perceptual(key, label, description):
    return Descriptor(
        key, "perceptual", label, description, None, 0.0, 1.0, "model",
        "HistGradientBoostingRegressor on the model input vector", model_input=False,
    )


def _research(key, label, description, unit, lo, hi, algorithm, value_type, **kw):
    return Descriptor(key, "research", label, description, unit, lo, hi, "dsp", algorithm, value_type=value_type, **kw)


SIGNAL_FAMILIES = {
    **dict.fromkeys(["spectral_centroid", "spectral_rolloff", "spectral_flux", "spectral_flatness",
                     "spectral_complexity", "spectral_contrast", "spectral_valleys", "energy_band_low",
                     "energy_band_mid_low", "energy_band_mid_high", "energy_band_high"], "spectral"),
    **dict.fromkeys(["zero_crossing_rate", "dissonance", "pitch_salience", "hfc"], "timbre"),
    **dict.fromkeys(["rms_mean", "rms_std", "loudness_range", "dynamic_complexity"], "dynamics"),
    **dict.fromkeys(["onset_rate", "beats_loudness", "tempo_stability", "bpm_second_peak_weight",
                     "danceability_dfa"], "rhythm"),
    **dict.fromkeys(["hpcp_entropy", "chords_changes_rate", "chords_number_rate", "tuning_frequency"], "tonal"),
}


def _signal(key, label, description, unit, lo, hi, algorithm, tier="advanced", **kw):
    return Descriptor(key, "signal", label, description, unit, lo, hi, "dsp", algorithm, tier=tier,
                      family=SIGNAL_FAMILIES[key], **kw)


DESCRIPTORS: list[Descriptor] = [
    # --- Group 1: core musical descriptors (full track) ---
    _musical("duration_ms", "Duration", "Length of the audio.", "ms", 0, None,
             "sample count / sample rate", model_input=False, value_type="int"),
    _musical("tempo", "Tempo", "Estimated tempo of the beat.", "BPM", 0, 250,
             "RhythmExtractor2013 (multifeature)"),
    _musical("key", "Key", "Estimated tonic (pitch class) of the track, spelled with sharps.", None, None, None,
             "KeyExtractor (edma profile)", model_input=False, value_type="string"),
    _musical("mode", "Mode", "Estimated mode of the key: major or minor.", None, None, None,
             "KeyExtractor (edma profile)", value_type="string"),
    _musical("key_confidence", "Key confidence", "Strength of the key estimate: correlation between the "
             "track's pitch-class profile and the chosen key profile.", None, 0.0, 1.0,
             "KeyExtractor (edma profile)"),
    _musical("loudness", "Loudness", "Integrated programme loudness per EBU R128 / ITU-R BS.1770.", "LUFS",
             -70.0, 10.0, "LoudnessEBUR128"),
    _musical("time_signature", "Time signature", "Estimated number of beats per bar (3, 4, 5 or 7).",
             "beats/bar", 3, 7,
             "BeatsLoudness + Beatogram + Meter", estimated=True, value_type="int"),
    # --- Group 2: perceptual descriptors (model predictions) ---
    _perceptual("energy", "Energy", "Perceived intensity and activity: fast, loud, dense and noisy music "
                "scores high, calm and sparse music low."),
    _perceptual("danceability", "Danceability", "How suitable the music is for dancing, driven by tempo, "
                "beat strength and rhythmic regularity."),
    _perceptual("valence", "Valence", "Musical positiveness: high values sound cheerful or euphoric, low "
                "values sad, tense or angry."),
    _perceptual("acousticness", "Acousticness", "Confidence that the recording is acoustic, i.e. made "
                "without electronic instruments or heavy production."),
    _perceptual("instrumentalness", "Instrumentalness", "Likelihood that the track contains no vocals. "
                "Values above 0.5 suggest an instrumental track."),
    _perceptual("liveness", "Liveness", "Likelihood that the recording was performed live in front of an "
                "audience. Values above 0.8 strongly suggest a live recording."),
    _perceptual("speechiness", "Speechiness", "Presence of spoken words. Above ~0.66 is mostly speech, "
                "0.33-0.66 mixes music and speech (e.g. rap), below 0.33 is mostly music."),
    # --- Group 3: research descriptors (full track) ---
    _research("chroma_vector", "Chroma", "Mean harmonic pitch-class profile, 12 bins from C to B, "
              "normalised so the strongest bin is 1.", None, 0.0, 1.0, "HPCP (12 bins)", "float[12]"),
    _research("beat_grid", "Beat grid", "Estimated beat positions.", "s", 0, None,
              "RhythmExtractor2013 (multifeature)", "float[]", model_input=False),
    _research("downbeats", "Downbeats", "Estimated first beat of each bar: every n-th beat (n = time "
              "signature), phase chosen by low-frequency beat loudness.", "s", 0, None,
              "BeatsLoudness + Meter", "float[]", estimated=True, model_input=False),
    _research("mfcc", "MFCC", "Mel-frequency cepstral coefficients 0-12 (mean and standard deviation over "
              "frames): a compact description of timbre.", None, None, None, "MFCC", "object"),
    # --- Group 4: signal descriptors (full track, frame means unless stated) ---
    _signal("spectral_centroid", "Brightness", "Spectral centroid: the spectrum's centre of mass. Higher "
            "means brighter, more treble-heavy sound.", "Hz", 0, 22050, "Centroid", tier="core"),
    _signal("spectral_rolloff", "Spectral rolloff", "Frequency below which 85% of the spectral energy lies.",
            "Hz", 0, 22050, "RollOff"),
    _signal("spectral_flux", "Spectral flux", "Frame-to-frame change of the spectrum (L2 norm).", None, 0, None,
            "Flux"),
    _signal("spectral_flatness", "Spectral flatness", "How noise-like (flat) rather than tonal (peaky) the "
            "spectrum is.", None, 0.0, 1.0, "FlatnessDB"),
    _signal("spectral_complexity", "Spectral complexity", "Number of peaks in the spectrum.", "peaks", 0, None,
            "SpectralComplexity"),
    _signal("spectral_contrast", "Spectral contrast", "Peak-to-valley contrast in 6 octave-based bands "
            "(low to high).", None, None, None, "SpectralContrast", value_type="float[6]"),
    _signal("spectral_valleys", "Spectral valleys", "Spectral valley level in the same 6 bands.", None, None,
            None, "SpectralContrast", value_type="float[6]"),
    _signal("energy_band_low", "Low band energy", "Share of spectral energy in 20-150 Hz.", None, 0.0, 1.0,
            "EnergyBandRatio"),
    _signal("energy_band_mid_low", "Mid-low band energy", "Share of spectral energy in 150-800 Hz.", None,
            0.0, 1.0, "EnergyBandRatio"),
    _signal("energy_band_mid_high", "Mid-high band energy", "Share of spectral energy in 800-4000 Hz.", None,
            0.0, 1.0, "EnergyBandRatio"),
    _signal("energy_band_high", "High band energy", "Share of spectral energy in 4-20 kHz.", None, 0.0, 1.0,
            "EnergyBandRatio"),
    _signal("zero_crossing_rate", "Zero-crossing rate", "Rate of sign changes of the waveform; high for noisy "
            "and percussive sound.", None, 0.0, 1.0, "ZeroCrossingRate"),
    _signal("dissonance", "Dissonance", "Sensory roughness of simultaneous spectral peaks.", None, 0.0, 1.0,
            "SpectralPeaks + Dissonance", tier="core"),
    _signal("pitch_salience", "Pitch salience", "How clearly a pitch is perceived (harmonic vs. inharmonic "
            "sound).", None, 0.0, 1.0, "PitchSalience"),
    _signal("hfc", "High-frequency content", "Energy weighted towards high frequencies; peaks at "
            "percussive attacks.", None, 0, None, "HFC"),
    _signal("rms_mean", "RMS level", "Mean root-mean-square amplitude of frames.", None, 0.0, 1.0, "RMS"),
    _signal("rms_std", "RMS variation", "Standard deviation of frame RMS amplitude.", None, 0.0, 1.0, "RMS"),
    _signal("loudness_range", "Dynamic range (LRA)", "EBU R128 loudness range: spread between quiet and loud "
            "passages.", "LU", 0, None, "LoudnessEBUR128", tier="core"),
    _signal("dynamic_complexity", "Dynamic complexity", "Average absolute deviation from the global loudness "
            "level.", "dB", 0, None, "DynamicComplexity"),
    _signal("onset_rate", "Note density", "Onsets (note or sound starts) per second.", "onsets/s", 0, None,
            "OnsetRate", tier="core"),
    _signal("beats_loudness", "Beat loudness", "Mean spectral energy at beat positions.", None, 0, None,
            "BeatsLoudness"),
    _signal("tempo_stability", "Tempo stability", "Weight of the main peak in the beat-interval histogram: "
            "1 means perfectly steady tempo.", None, 0.0, 1.0, "BpmHistogramDescriptors", tier="core"),
    _signal("bpm_second_peak_weight", "Secondary tempo weight", "Weight of the second peak in the "
            "beat-interval histogram.", None, 0.0, 1.0, "BpmHistogramDescriptors"),
    _signal("danceability_dfa", "Rhythmic regularity (DFA)", "Essentia's detrended-fluctuation danceability "
            "value (roughly 0-3). A DSP input to the model, not the learned danceability.", None, 0.0, None,
            "Danceability"),
    _signal("hpcp_entropy", "Chroma entropy", "Entropy of the pitch-class profile per frame; high for "
            "harmonically diffuse or noisy music.", "bits", 0.0, 3.585, "HPCP + Entropy"),
    _signal("chords_changes_rate", "Chord change rate", "Share of analysis windows where the estimated chord "
            "changes.", None, 0.0, 1.0, "ChordsDetection + ChordsDescriptors"),
    _signal("chords_number_rate", "Chord variety", "Number of distinct frequent chords relative to the "
            "number of chord estimates.", None, 0.0, 1.0, "ChordsDetection + ChordsDescriptors"),
    _signal("tuning_frequency", "Tuning frequency", "Estimated reference frequency of A4.", "Hz", 400.0, 480.0,
            "TuningFrequency"),
]

BY_KEY = {d.key: d for d in DESCRIPTORS}


def keys_in_group(group: str) -> list[str]:
    return [d.key for d in DESCRIPTORS if d.group == group]
