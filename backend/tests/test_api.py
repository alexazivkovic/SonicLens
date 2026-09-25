"""API behaviour: schemas, caching, versioning, errors, temp-file cleanup, insert-only database."""

import asyncio
import hashlib
import io

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select

import app.api.service
from app.api.schemas import AnalysisResponse, DescriptorInfo, HealthResponse, ModelCard
from app.core.config import BACKEND_DIR, Settings
from app.db import cache
from app.db.models import Analysis
from app.extraction.registry import DESCRIPTORS, keys_in_group
from app.factory import create_app
from tests.conftest import make_bundle
from tests.synth import SR, click_track, sine, write_wav


def wav_bytes(tmp_path, audio: np.ndarray, name: str) -> bytes:
    return write_wav(tmp_path / name, audio).read_bytes()


@pytest.fixture
def settings(tmp_path, dummy_model_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        model_path=dummy_model_path,
        model_card_path=BACKEND_DIR / "models" / "model_card.json",
        temp_dir=tmp_path / "uploads",
        max_upload_mb=5,
        cors_origins=["http://localhost:5173"],
    )


@pytest.fixture
def statements(settings):
    """Every SQL statement the app executes (checked for insert-only behaviour)."""
    return []


@pytest.fixture
def client(settings, statements):
    api = create_app(settings)
    engine = api.state.sessions.kw["bind"]
    event.listen(engine, "before_cursor_execute", lambda *args: statements.append(args[2]))
    with TestClient(api) as c:
        yield c
    # Audio is never kept: the upload directory is empty after every test.
    assert not list(settings.temp_dir.glob("*")) if settings.temp_dir.exists() else True
    assert not any(s.lstrip().upper().startswith(("UPDATE", "DELETE")) for s in statements)


@pytest.fixture
def song(tmp_path):
    return wav_bytes(tmp_path, click_track(120, 4, duration=6.0), "song.wav")


def upload(client, data: bytes, name: str = "song.wav"):
    return client.post("/analyze", files={"file": (name, io.BytesIO(data), "application/octet-stream")})


def row_count(api) -> int:
    with api.state.sessions() as s:
        return s.scalar(select(func.count()).select_from(Analysis))


# --- Analyze & cache ------------------------------------------------------------------------


def test_analyze_then_cache_hit(client, song):
    first = upload(client, song)
    assert first.status_code == 200, first.text
    body = AnalysisResponse.model_validate(first.json())
    assert body.cached is False
    assert body.audio_hash == hashlib.sha256(song).hexdigest()
    assert body.musical.duration_ms == 6000
    assert body.perceptual.energy == pytest.approx(0.3)  # dummy model
    assert "model-test-0" in body.pipeline_version

    second = upload(client, song, name="renamed.wav")
    assert second.status_code == 200
    again = AnalysisResponse.model_validate(second.json())
    assert again.cached is True
    assert again.created_at == body.created_at
    ignore = {"cached", "processing_ms"}
    assert again.model_dump(exclude=ignore) == body.model_dump(exclude=ignore)
    assert row_count(client.app) == 1


def test_different_file_is_a_cache_miss(client, song, tmp_path):
    other = wav_bytes(tmp_path, click_track(100, 4, duration=6.0), "other.wav")
    assert upload(client, song).json()["cached"] is False
    assert upload(client, other).json()["cached"] is False
    assert row_count(client.app) == 2


def test_changed_pipeline_version_inserts_new_row_and_keeps_old(settings, song, tmp_path):
    with TestClient(create_app(settings)) as c:
        old = upload(c, song).json()

    new_model = tmp_path / "model_v1.joblib"
    joblib.dump(make_bundle(constant=0.6, model_version="test-1"), new_model)
    new_settings = Settings(**{**settings.__dict__, "model_path": new_model})
    api = create_app(new_settings)
    with TestClient(api) as c:
        new = upload(c, song).json()
        assert new["cached"] is False
        assert new["pipeline_version"] != old["pipeline_version"]
        assert new["perceptual"]["energy"] == pytest.approx(0.6)
        assert upload(c, song).json()["cached"] is True

    assert row_count(api) == 2
    with api.state.sessions() as s:
        stored = cache.lookup(s, old["audio_hash"], old["pipeline_version"])
    assert stored is not None and stored.features["perceptual"]["energy"] == pytest.approx(0.3)


def test_get_by_hash(client, song):
    posted = upload(client, song).json()
    got = client.get(f"/analyze/{posted['audio_hash']}")
    assert got.status_code == 200
    body = AnalysisResponse.model_validate(got.json())
    assert body.cached is True
    assert body.musical == AnalysisResponse.model_validate(posted).musical


def test_get_by_hash_errors(client):
    assert client.get(f"/analyze/{'0' * 64}").status_code == 404
    assert client.get("/analyze/not-a-hash").status_code == 422


def test_concurrent_duplicate_insert_returns_existing_row(settings):
    sessions = cache.make_sessionmaker(settings.database_url)
    with sessions() as s:
        first, inserted_first = cache.insert(s, "a" * 64, "v1", {"x": 1})
    with sessions() as s:
        second, inserted_second = cache.insert(s, "a" * 64, "v1", {"x": 2})
    assert inserted_first and not inserted_second
    assert second.id == first.id and second.features == {"x": 1}


# --- Upload errors --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make, name, status, message",
    [
        (lambda tmp: b"plain text", "notes.txt", 415, "Unsupported file type"),
        (lambda tmp: b"ID3" + b"\x00garbage" * 500, "broken.mp3", 422, "could not be decoded"),
        (lambda tmp: b"", "empty.wav", 422, "could not be decoded"),
        (lambda tmp: wav_bytes(tmp, sine(440.0, 1.5), "short.wav"), "short.wav", 422, "at least 3 s"),
        (lambda tmp: wav_bytes(tmp, np.zeros(5 * SR, dtype=np.float32), "silent.wav"), "silent.wav", 422, "silent"),
    ],
)
def test_bad_uploads(client, tmp_path, make, name, status, message):
    r = upload(client, make(tmp_path), name=name)
    assert r.status_code == status, r.text
    assert message in r.json()["detail"]
    assert row_count(client.app) == 0


def test_upload_limit(settings, tmp_path):
    small = Settings(**{**settings.__dict__, "max_upload_mb": 0.05})
    with TestClient(create_app(small)) as c:
        r = upload(c, wav_bytes(tmp_path, sine(440.0, 5.0), "big.wav"))  # ~430 KB
        assert r.status_code == 413
        assert "limit" in r.json()["detail"]
    assert not list(small.temp_dir.glob("*"))


def test_streaming_limit_without_content_length(tmp_path):
    # Chunked uploads carry no Content-Length; save_upload enforces the limit while streaming.
    from starlette.datastructures import UploadFile

    upload_dir = tmp_path / "uploads"
    big = UploadFile(file=io.BytesIO(b"x" * 300_000), filename="big.wav")
    with pytest.raises(app.api.service.UploadTooLargeError):
        asyncio.run(app.api.service.save_upload(big, upload_dir, ".wav", max_bytes=100_000))
    assert not list(upload_dir.glob("*"))


# --- Model availability ----------------------------------------------------------------------


@pytest.mark.parametrize("problem", ["missing", "mismatched"])
def test_without_a_usable_model(settings, tmp_path, song, problem):
    path = tmp_path / "bad.joblib"
    if problem == "mismatched":
        joblib.dump(make_bundle(extractor_version="0.0.0"), path)
    broken = Settings(**{**settings.__dict__, "model_path": path})
    with TestClient(create_app(broken)) as c:
        health = HealthResponse.model_validate(c.get("/health").json())
        assert health.status == "degraded" and not health.model_loaded and health.model_error
        r = upload(c, song)
        assert r.status_code == 503
        assert "not available" in r.json()["detail"]
        assert c.get("/features").status_code == 200


# --- Metadata endpoints ------------------------------------------------------------------------


def test_health(client):
    body = HealthResponse.model_validate(client.get("/health").json())
    assert body.status == "ok" and body.essentia_loaded and body.model_loaded
    assert body.model_version == "test-0"


def test_features(client):
    items = [DescriptorInfo.model_validate(d) for d in client.get("/features").json()]
    assert [d.key for d in items] == [d.key for d in DESCRIPTORS]
    core_signal = {d.key for d in items if d.group == "signal" and d.tier == "core"}
    assert core_signal == {"spectral_centroid", "loudness_range", "onset_rate", "tempo_stability", "dissonance"}
    assert {d.key for d in items if d.method == "model"} == set(keys_in_group("perceptual"))
    assert {d.key for d in items if d.estimated} == {"time_signature", "downbeats"}


def test_model_card(client):
    card = ModelCard.model_validate(client.get("/model").json())
    assert set(card.targets) == set(keys_in_group("perceptual"))


def test_response_schema_matches_registry():
    for group, schema in [("musical", "MusicalDescriptors"), ("perceptual", "PerceptualDescriptors"),
                          ("research", "ResearchDescriptors"), ("signal", "SignalDescriptors")]:
        fields = list(getattr(__import__("app.api.schemas", fromlist=[schema]), schema).model_fields)
        assert fields == keys_in_group(group), group


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    assert "/analyze" in client.get("/openapi.json").json()["paths"]


def test_cors_preflight(client):
    r = client.options("/analyze", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
