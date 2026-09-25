import { useEffect, useRef, useState } from 'react'
import { ApiError, analyze, checkFile, getFeatures, getHealth, getModelCard } from './api'
import type { Analysis, DescriptorInfo, Health, ModelCard } from './types'
import { About } from './components/About'
import { AdvancedSection } from './components/Advanced'
import { Overview } from './components/Overview'
import { ResearchSection } from './components/Research'
import { StatusBar } from './components/StatusBar'
import { UploadZone } from './components/UploadZone'

type ApiState = { kind: 'loading' } | { kind: 'down'; message: string } | { kind: 'up'; health: Health }

/** Seconds since `startedAt` (a performance.now() timestamp), ticking while `running`. */
function useElapsed(running: boolean, startedAt: number): number {
  const [now, setNow] = useState(0)
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => setNow(performance.now()), 100)
    return () => clearInterval(id)
  }, [running])
  return Math.max(0, (now - startedAt) / 1000)
}

function downloadJson(result: Analysis) {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `soniclens-${result.audio_hash.slice(0, 12)}.json`
  a.click()
  URL.revokeObjectURL(url)
}

export default function App() {
  const [api, setApi] = useState<ApiState>({ kind: 'loading' })
  const [features, setFeatures] = useState<DescriptorInfo[]>([])
  const [model, setModel] = useState<ModelCard | null>(null)
  // A result and the name of the file it belongs to always change together.
  const [result, setResult] = useState<{ analysis: Analysis; fileName: string } | null>(null)
  const [pendingName, setPendingName] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAbout, setShowAbout] = useState(false)
  const controller = useRef<AbortController | null>(null)
  const [startedAt, setStartedAt] = useState(0)
  const elapsed = useElapsed(busy, startedAt)

  useEffect(() => {
    Promise.all([getHealth(), getFeatures()])
      .then(([health, feats]) => {
        setApi({ kind: 'up', health })
        setFeatures(feats)
      })
      .catch((e: Error) => setApi({ kind: 'down', message: e.message }))
    getModelCard().then(setModel).catch(() => setModel(null))
  }, [])

  const onFile = async (file: File) => {
    const problem = checkFile(file)
    if (problem) {
      setError(problem)
      return
    }
    controller.current?.abort()
    controller.current = new AbortController()
    setStartedAt(performance.now())
    setBusy(true)
    setError(null)
    setPendingName(file.name)
    try {
      setResult({ analysis: await analyze(file, controller.current.signal), fileName: file.name })
    } catch (e) {
      if ((e as Error).name === 'AbortError') return
      setError(e instanceof ApiError ? e.message : `Unexpected error: ${(e as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <img src="/favicon.svg" alt="" width={28} height={28} />
          <div>
            <h1>SonicLens</h1>
            <p className="muted small">Low-level and perceptual audio descriptors</p>
          </div>
        </div>
        <div className="header-actions">
          <ApiStatus api={api} />
          <button type="button" className="button" onClick={() => setShowAbout((s) => !s)} aria-expanded={showAbout}>
            About
          </button>
        </div>
      </header>

      {showAbout && <About model={model} onClose={() => setShowAbout(false)} />}

      <main>
        <UploadZone busy={busy} compact={result !== null} onFile={onFile} />

        {busy && (
          <p className="progress" role="status">
            <span className="spinner" aria-hidden="true" /> Analysing <strong>{pendingName}</strong>… {elapsed.toFixed(1)} s
            <span className="muted small"> (a full song takes a few seconds)</span>
          </p>
        )}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}

        {result && (
          // Keyed by hash so per-result UI state (raw JSON toggle, timeline position) resets.
          <Results
            key={result.analysis.audio_hash}
            result={result.analysis}
            fileName={result.fileName}
            stale={busy}
            features={features}
            model={model}
          />
        )}
      </main>

      <footer className="page-footer muted small">
        SonicLens · open source under AGPL-3.0 · trained on the Free Music Archive (CC BY 4.0) · audio is never stored
      </footer>
    </div>
  )
}

interface ResultsProps {
  result: Analysis
  fileName: string
  stale: boolean
  features: DescriptorInfo[]
  model: ModelCard | null
}

function Results({ result, fileName, stale, features, model }: ResultsProps) {
  const [showJson, setShowJson] = useState(false)
  return (
    <div className={stale ? 'results stale' : 'results'}>
      <div className="result-head">
        <h2 className="file-name">{fileName}</h2>
        <div className="result-actions">
          <button type="button" className="button" onClick={() => setShowJson((s) => !s)} aria-expanded={showJson}>
            {showJson ? 'Hide raw JSON' : 'Show raw JSON'}
          </button>
          <button type="button" className="button" onClick={() => downloadJson(result)}>
            Download JSON
          </button>
        </div>
      </div>
      {showJson && <pre className="raw-json">{JSON.stringify(result, null, 2)}</pre>}
      <Overview result={result} features={features} model={model} />
      <ResearchSection result={result} />
      <AdvancedSection result={result} features={features} />
      <StatusBar result={result} />
    </div>
  )
}

function ApiStatus({ api }: { api: ApiState }) {
  if (api.kind === 'loading') return <span className="api-status">Connecting…</span>
  if (api.kind === 'down')
    return (
      <span className="api-status down" title={api.message}>
        <span className="dot" aria-hidden="true" /> API unreachable
      </span>
    )
  const { health } = api
  return health.status === 'ok' ? (
    <span className="api-status ok" title={health.pipeline_version ?? ''}>
      <span className="dot" aria-hidden="true" /> API ready
    </span>
  ) : (
    <span className="api-status degraded" title={health.model_error ?? ''}>
      <span className="dot" aria-hidden="true" /> API degraded: {health.model_loaded ? 'Essentia unavailable' : 'model not loaded'}
    </span>
  )
}
