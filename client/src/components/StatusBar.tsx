import { useState } from 'react'
import type { Analysis } from '../types'
import { formatDateTime } from '../format'

export function StatusBar({ result }: { result: Analysis }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(result.audio_hash)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable (e.g. insecure context); the full hash is in the title attribute */
    }
  }
  return (
    <footer className="status" aria-label="Result status">
      <span>
        {result.cached ? (
          <>
            <strong>Served from cache</strong> (computed {formatDateTime(result.created_at)})
          </>
        ) : (
          <>
            <strong>Freshly computed</strong> in {(result.processing_ms / 1000).toFixed(1)} s ({formatDateTime(result.created_at)})
          </>
        )}
      </span>
      <span>
        Pipeline <code>{result.pipeline_version}</code>
      </span>
      <span>
        Audio hash{' '}
        <code title={result.audio_hash}>
          {result.audio_hash.slice(0, 12)}…
        </code>{' '}
        <button type="button" className="link-button" onClick={copy} aria-label="Copy the full audio hash">
          {copied ? 'Copied' : 'Copy'}
        </button>
      </span>
    </footer>
  )
}
