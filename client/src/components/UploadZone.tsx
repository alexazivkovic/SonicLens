import { useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_MB } from '../api'

interface Props {
  busy: boolean
  compact: boolean
  onFile: (file: File) => void
}

export function UploadZone({ busy, compact, onFile }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)

  const drop = (e: DragEvent) => {
    e.preventDefault()
    setOver(false)
    const file = e.dataTransfer.files[0]
    if (file && !busy) onFile(file)
  }

  return (
    <div
      className={`dropzone${over ? ' over' : ''}${compact ? ' compact' : ''}${busy ? ' busy' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
      data-testid="dropzone"
    >
      <input
        ref={input}
        type="file"
        accept={ACCEPTED_EXTENSIONS.join(',')}
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
          e.target.value = ''
        }}
      />
      <p className="dropzone-title">{compact ? 'Analyse another file' : 'Drop an audio file here'}</p>
      <p className="dropzone-hint">
        {ACCEPTED_EXTENSIONS.join(', ')} · up to {MAX_UPLOAD_MB} MB · the audio is analysed and deleted, never stored
      </p>
      <button type="button" className="button primary" disabled={busy} onClick={() => input.current?.click()}>
        Choose file
      </button>
    </div>
  )
}
