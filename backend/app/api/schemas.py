"""Pydantic request/response models. Field names mirror app.extraction.registry."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Unit = Field(ge=0.0, le=1.0)


class MusicalDescriptors(BaseModel):
    duration_ms: int = Field(ge=0, description="Length of the audio in milliseconds.")
    tempo: float = Field(description="Estimated tempo in BPM.")
    key: str = Field(description="Estimated tonic, spelled with sharps (C, C#, ... B).")
    mode: Literal["major", "minor"]
    key_confidence: float = Unit
    loudness: float = Field(description="Integrated loudness, LUFS (EBU R128).")
    time_signature: int = Field(description="Estimated beats per bar (estimated).")


class PerceptualDescriptors(BaseModel):
    energy: float = Unit
    danceability: float = Unit
    valence: float = Unit
    acousticness: float = Unit
    instrumentalness: float = Unit
    liveness: float = Unit
    speechiness: float = Unit


class MfccStats(BaseModel):
    mean: list[float] = Field(min_length=13, max_length=13)
    std: list[float] = Field(min_length=13, max_length=13)


class ResearchDescriptors(BaseModel):
    chroma_vector: list[float] = Field(min_length=12, max_length=12, description="Mean HPCP, C..B, max = 1.")
    beat_grid: list[float] = Field(description="Beat positions in seconds.")
    downbeats: list[float] = Field(description="Estimated downbeat positions in seconds (estimated).")
    mfcc: MfccStats


class SignalDescriptors(BaseModel):
    spectral_centroid: float
    spectral_rolloff: float
    spectral_flux: float
    spectral_flatness: float
    spectral_complexity: float
    spectral_contrast: list[float] = Field(min_length=6, max_length=6)
    spectral_valleys: list[float] = Field(min_length=6, max_length=6)
    energy_band_low: float
    energy_band_mid_low: float
    energy_band_mid_high: float
    energy_band_high: float
    zero_crossing_rate: float
    dissonance: float
    pitch_salience: float
    hfc: float
    rms_mean: float
    rms_std: float
    loudness_range: float
    dynamic_complexity: float
    onset_rate: float
    beats_loudness: float
    tempo_stability: float
    bpm_second_peak_weight: float
    danceability_dfa: float
    hpcp_entropy: float
    chords_changes_rate: float
    chords_number_rate: float
    tuning_frequency: float


class ModelWindow(BaseModel):
    start_s: float
    end_s: float


class ModelInput(BaseModel):
    window: ModelWindow = Field(description="The part of the track the model input vector was computed on.")
    features: dict[str, float] = Field(description="The ordered, named model input vector.")


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    audio_hash: str = Field(description="SHA-256 of the uploaded bytes.")
    pipeline_version: str
    cached: bool = Field(description="True if this result was served from the cache.")
    created_at: datetime = Field(description="When this result was first computed (UTC).")
    processing_ms: int = Field(description="Time spent on this request.")
    extractor_version: str
    essentia_version: str
    model_version: str
    musical: MusicalDescriptors
    perceptual: PerceptualDescriptors
    research: ResearchDescriptors
    signal: SignalDescriptors
    model_input: ModelInput


class DescriptorInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    key: str
    group: Literal["musical", "perceptual", "research", "signal"]
    label: str
    description: str
    unit: str | None
    min: float | None
    max: float | None
    method: Literal["dsp", "model"]
    algorithm: str
    tier: Literal["core", "advanced"]
    estimated: bool
    model_input: bool
    value_type: str
    family: Literal["spectral", "timbre", "dynamics", "rhythm", "tonal"] | None = Field(
        default=None, description="Descriptor family (signal descriptors only)."
    )


class Metrics(BaseModel):
    mae: float
    rmse: float
    r2: float
    spearman: float
    r2_folds: list[float] | None = None


class TargetCard(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    hyperparameters: dict
    cv: Metrics
    test: Metrics
    baselines: dict


class ModelCard(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    model_version: str
    created: str
    extractor_version: str
    essentia_version: str
    sklearn_version: str
    description: str
    training_data: dict
    features: dict
    evaluation: dict
    targets: dict[str, TargetCard]
    tempo_validation: dict
    limitations: list[str]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: Literal["ok", "degraded"]
    essentia_loaded: bool
    essentia_version: str | None
    model_loaded: bool
    model_version: str | None
    model_error: str | None
    extractor_version: str
    pipeline_version: str | None


class ErrorResponse(BaseModel):
    detail: str
