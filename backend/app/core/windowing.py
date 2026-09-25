"""The analysis window used for model input features.

FMA's 30 s clips were cut by `creation.py` as `start = duration // 2 - 15` (whole seconds),
and tracks of 30 s or less were copied whole. Model inputs at inference are computed on the
same window so the model sees audio positioned like its training clips.
"""

MODEL_WINDOW_S = 30

# Tracks shorter than this are analysed whole. FMA clips are nominally 30 s but decode to
# slightly more or less; trimming a fraction of a second would change nothing but cost a
# second analysis pass.
_WHOLE_TRACK_BELOW_S = MODEL_WINDOW_S + 1


def model_window(n_samples: int, sr: int) -> tuple[int, int]:
    """Return the (start, end) sample indices of the model input window."""
    duration_s = n_samples / sr
    if duration_s < _WHOLE_TRACK_BELOW_S:
        return 0, n_samples
    start_s = int(duration_s) // 2 - MODEL_WINDOW_S // 2
    start = start_s * sr
    return start, min(start + MODEL_WINDOW_S * sr, n_samples)
