"""Audio decoding and normalisation to the analysis format (44.1 kHz float32)."""

from pathlib import Path

import essentia
import essentia.standard as es
import numpy as np

from app.core.errors import AudioDecodeError, AudioTooShortError, SilentAudioError

essentia.log.infoActive = False

SAMPLE_RATE = 44100
MIN_DURATION_S = 3.0
# Peak amplitude below which a signal counts as silent (-80 dBFS).
SILENCE_PEAK = 1e-4


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode an audio file (mp3, wav, flac, ogg, m4a, ...).

    Returns (samples, sample_rate), samples shaped (n_samples, n_channels), float32.
    """
    try:
        audio, sr, n_channels, *_ = es.AudioLoader(filename=str(path))()
    except RuntimeError as exc:
        raise AudioDecodeError(f"Could not decode audio: {exc}") from exc
    if audio.size == 0:
        raise AudioDecodeError("The file contains no audio samples.")
    # AudioLoader always returns two channels; keep one for mono sources.
    audio = audio[:, :1] if n_channels == 1 else audio
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return x
    return es.Resample(inputSampleRate=sr, outputSampleRate=SAMPLE_RATE, quality=1)(x)


def prepare(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate audio and convert it to the analysis format.

    `audio` is 1-D (mono) or 2-D shaped (n_samples, n_channels), at any sample rate.
    Returns (mono, stereo) at SAMPLE_RATE: mono is the channel average, stereo is shaped
    (n, 2) and is only used for EBU R128 loudness. Mono sources become dual-mono stereo,
    and sources with more than two channels are reduced to their downmix.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[1] < 1:
        raise AudioDecodeError("Audio must be 1-D or shaped (n_samples, n_channels).")
    if not np.all(np.isfinite(audio)):
        raise AudioDecodeError("Audio contains non-finite sample values.")
    if audio.shape[0] / sr < MIN_DURATION_S:
        raise AudioTooShortError(
            f"Audio is {audio.shape[0] / sr:.2f} s long; at least {MIN_DURATION_S:.0f} s is required."
        )
    if np.max(np.abs(audio)) < SILENCE_PEAK:
        raise SilentAudioError("The audio is silent (peak level below -80 dBFS).")

    mono = _resample(np.ascontiguousarray(audio.mean(axis=1)), sr)
    if audio.shape[1] == 2:
        left = _resample(np.ascontiguousarray(audio[:, 0]), sr)
        right = _resample(np.ascontiguousarray(audio[:, 1]), sr)
        stereo = np.stack([left, right], axis=1)
    else:
        stereo = np.stack([mono, mono], axis=1)
    return mono.astype(np.float32), np.ascontiguousarray(stereo, dtype=np.float32)
