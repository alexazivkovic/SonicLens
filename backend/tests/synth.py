"""Synthetic test signals with known properties."""

import wave
from pathlib import Path

import numpy as np

SR = 44100


def sine(freq: float, duration: float, amplitude: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(round(duration * sr))) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def click_track(bpm: float, beats_per_bar: int, duration: float = 30.0, sr: int = SR) -> np.ndarray:
    """Clicks at `bpm`. The first beat of each bar (starting at t=0) is a loud low thump,
    the other beats are quieter high clicks."""
    x = np.zeros(int(duration * sr), dtype=np.float32)
    t = np.arange(int(0.03 * sr)) / sr
    env = np.exp(-t * 80)
    period = 60.0 / bpm
    i = 0
    while i * period < duration - 0.1:
        start = int(round(i * period * sr))
        strong = i % beats_per_bar == 0
        freq, amp = (60.0, 1.0) if strong else (1500.0, 0.35)
        x[start : start + len(t)] += (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        i += 1
    return 0.8 * x / np.max(np.abs(x))


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> Path:
    audio = audio if audio.ndim == 2 else audio[:, None]
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(audio.shape[1])
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())
    return path
