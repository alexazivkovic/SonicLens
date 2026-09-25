"""Compare Essentia KeyExtractor profiles on synthesised tonal audio in all 24 keys.

FMA has no key annotations, so this uses synthetic cadences with known keys. Each test piece
is a looped four-chord progression of harmonic complex tones with a scale melody on top.
Progressions include classical cadences (I-IV-V-I, i-iv-V-i) and the pop/electronic patterns
that make key finding hard (I-V-vi-IV, vi-IV-I-V, Aeolian i-VI-III-VII and i-VII-VI-VII with no
leading tone). Each is rendered with several timbres, with and without detuning (+30 cents) and
noise percussion. The score is the MIREX weighted key score (correct 1.0, fifth 0.5,
relative 0.3, parallel 0.2).

Writes reports/key_profiles.json. Usage (from backend/):  python -m training.eval_key_profiles
"""

import json
from pathlib import Path

import essentia
import essentia.standard as es
import numpy as np

essentia.log.infoActive = False

SR = 44100
PROFILES = ["edma", "temperley", "krumhansl"]
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
REPORT = Path(__file__).resolve().parents[1] / "reports" / "key_profiles.json"


def _tone(midi: float, dur: float, n_harm: int, decay: float) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    x = sum((decay**h) * np.sin(2 * np.pi * f0 * (h + 1) * t) for h in range(n_harm) if f0 * (h + 1) < SR / 2)
    env = np.minimum(1, t / 0.01) * np.exp(-t * 1.5)
    return x * env


def _triad(root: int, quality: str) -> tuple:
    return (root, root + (4 if quality == "M" else 3), root + 7)


# Chord roots (semitones above the tonic) and qualities for each progression.
PROGRESSIONS = {
    "major": {
        "I-IV-V-I": [(0, "M"), (5, "M"), (7, "M"), (0, "M")],
        "I-V-vi-IV": [(0, "M"), (7, "M"), (9, "m"), (5, "M")],
        "vi-IV-I-V": [(9, "m"), (5, "M"), (0, "M"), (7, "M")],
    },
    "minor": {
        "i-iv-V-i": [(0, "m"), (5, "m"), (7, "M"), (0, "m")],
        "i-VI-III-VII": [(0, "m"), (8, "M"), (3, "M"), (10, "M")],
        "i-VII-VI-VII": [(0, "m"), (10, "M"), (8, "M"), (10, "M")],
    },
}
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}


def render_piece(tonic, mode, progression, n_harm, decay, detune_cents, drums, rng) -> np.ndarray:
    scale = SCALES[mode]
    base = 60 + tonic + detune_cents / 100
    beat = 0.5
    out = []
    for i, (root, quality) in enumerate(PROGRESSIONS[mode][progression]):
        seg = np.zeros(int(4 * beat * SR))
        for off in _triad(root, quality):
            seg += _tone(base + off, 4 * beat, n_harm, decay)
        seg += 0.6 * _tone(base - 12 + root, 4 * beat, n_harm, decay)
        for j in range(4):
            melody = base + 12 + scale[(2 * i + j) % 7]
            s = int(j * beat * SR)
            seg[s : s + int(beat * SR)] += 0.8 * _tone(melody, beat, n_harm, decay)
        out.append(seg)
    x = np.tile(np.concatenate(out), 3)
    x = x / np.max(np.abs(x))
    if drums:
        hit = int(0.05 * SR)
        env = np.exp(-np.arange(hit) / SR * 60)
        for s in range(0, len(x) - hit, int(beat * SR / 2)):
            x[s : s + hit] += 0.5 * env * rng.standard_normal(hit)
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


def mirex_score(est_key: str, est_scale: str, tonic: int, minor: bool) -> float:
    est = NOTES.index(ENHARMONIC.get(est_key, est_key))
    est_minor = est_scale == "minor"
    if est == tonic and est_minor == minor:
        return 1.0
    if est_minor == minor and (est - tonic) % 12 in (5, 7):
        return 0.5
    if est_minor != minor and (est - tonic) % 12 == (9 if not minor else 3):
        return 0.3
    if est == tonic:
        return 0.2
    return 0.0


def main() -> None:
    rng = np.random.default_rng(0)
    timbres = [(1, 0.0), (4, 0.6), (8, 0.8)]
    results = {p: {"scores": [], "by_progression": {}} for p in PROFILES}
    n = 0
    for tonic in range(12):
        for mode, progressions in PROGRESSIONS.items():
            for progression in progressions:
                for n_harm, decay in timbres:
                    for detune in (0, 30):
                        for drums in (False, True):
                            audio = render_piece(tonic, mode, progression, n_harm, decay, detune, drums, rng)
                            n += 1
                            for profile in PROFILES:
                                key, scale, _ = es.KeyExtractor(profileType=profile, sampleRate=SR)(audio)
                                score = mirex_score(key, scale, tonic, mode == "minor")
                                results[profile]["scores"].append(score)
                                results[profile]["by_progression"].setdefault(progression, []).append(score)

    summary = {
        "n_pieces": n,
        "method": " ".join(__doc__.strip().split("\n\n")[:2]).replace("\n", " "),
        "profiles": {
            p: {
                "accuracy": float(np.mean([sc == 1.0 for sc in r["scores"]])),
                "mirex_weighted": float(np.mean(r["scores"])),
                "mirex_by_progression": {k: float(np.mean(v)) for k, v in r["by_progression"].items()},
            }
            for p, r in results.items()
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["profiles"], indent=2))


if __name__ == "__main__":
    main()
