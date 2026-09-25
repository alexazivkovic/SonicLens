// Mirrors the backend's Pydantic models (backend/app/api/schemas.py).

export type PerceptualKey =
  | 'energy'
  | 'danceability'
  | 'valence'
  | 'acousticness'
  | 'instrumentalness'
  | 'liveness'
  | 'speechiness'

export interface Musical {
  duration_ms: number
  tempo: number
  key: string
  mode: 'major' | 'minor'
  key_confidence: number
  loudness: number
  time_signature: number
}

export interface Research {
  chroma_vector: number[]
  beat_grid: number[]
  downbeats: number[]
  mfcc: { mean: number[]; std: number[] }
}

export type SignalValue = number | number[]

export interface Analysis {
  audio_hash: string
  pipeline_version: string
  cached: boolean
  created_at: string
  processing_ms: number
  extractor_version: string
  essentia_version: string
  model_version: string
  musical: Musical
  perceptual: Record<PerceptualKey, number>
  research: Research
  signal: Record<string, SignalValue>
  model_input: {
    window: { start_s: number; end_s: number }
    features: Record<string, number>
  }
}

export type Group = 'musical' | 'perceptual' | 'research' | 'signal'
export type Family = 'spectral' | 'timbre' | 'dynamics' | 'rhythm' | 'tonal'

export interface DescriptorInfo {
  key: string
  group: Group
  label: string
  description: string
  unit: string | null
  min: number | null
  max: number | null
  method: 'dsp' | 'model'
  algorithm: string
  tier: 'core' | 'advanced'
  estimated: boolean
  model_input: boolean
  value_type: string
  family: Family | null
}

export interface Metrics {
  mae: number
  rmse: number
  r2: number
  spearman: number
}

export interface ModelCard {
  model_version: string
  created: string
  description: string
  training_data: {
    dataset: string
    citation: string
    metadata_license: string
    targets_source: string
    n_tracks_used: number
    n_dev: number
    n_test: number
    n_artists_test: number
  }
  targets: Record<PerceptualKey, { cv: Metrics; test: Metrics }>
  limitations: string[]
}

export interface Health {
  status: 'ok' | 'degraded'
  essentia_loaded: boolean
  model_loaded: boolean
  model_version: string | null
  model_error: string | null
  pipeline_version: string | null
}
