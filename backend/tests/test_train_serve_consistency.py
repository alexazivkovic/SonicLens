"""The training pipeline must compute features with exactly the code the app uses."""

import training.extract
from app.core.audio import load_audio
from app.extraction import extract_features
from tests.synth import click_track, write_wav


def test_training_row_equals_direct_extraction(tmp_path, monkeypatch):
    path = write_wav(tmp_path / "clip.wav", click_track(120, 4, duration=10.0))
    monkeypatch.setattr(training.extract, "audio_path", lambda track_id: path)

    row = training.extract.process((123, 10.0))
    direct = extract_features(*load_audio(path))

    assert row["status"] == "ok"
    assert {k: row[k] for k in direct["model_input"]["features"]} == direct["model_input"]["features"]


def test_training_skips_truncated_clips(tmp_path, monkeypatch):
    # The track is 200 s long, so its clip should be ~30 s; a 10 s clip is truncated.
    path = write_wav(tmp_path / "clip.wav", click_track(120, 4, duration=10.0))
    monkeypatch.setattr(training.extract, "audio_path", lambda track_id: path)
    assert training.extract.process((123, 200.0))["status"] == "truncated"


def test_training_skips_too_short_clips(tmp_path, monkeypatch):
    path = write_wav(tmp_path / "clip.wav", click_track(120, 4, duration=2.0))
    monkeypatch.setattr(training.extract, "audio_path", lambda track_id: path)
    assert training.extract.process((123, 2.0))["status"] == "AudioTooShortError"
