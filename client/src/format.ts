export const PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

export function formatDuration(ms: number): string {
  const total = Math.round(ms / 1000)
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

export function formatSeconds(s: number): string {
  return s >= 60 ? `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}` : `${s.toFixed(2)} s`
}

/** A readable number: few decimals for large values, more for small ones. */
export function formatNumber(v: number): string {
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1000) return v.toLocaleString('en', { maximumFractionDigits: 0 })
  if (a >= 100) return v.toFixed(1)
  if (a >= 1) return v.toFixed(2)
  if (a >= 0.01) return v.toFixed(3)
  return v.toExponential(2)
}

export function formatValue(v: number | number[]): string {
  return Array.isArray(v) ? v.map(formatNumber).join(', ') : formatNumber(v)
}

/** How far a learned descriptor can be trusted, from its held-out test R². */
export function reliability(r2: number): 'good' | 'moderate' | 'low' {
  if (r2 >= 0.6) return 'good'
  if (r2 >= 0.4) return 'moderate'
  return 'low'
}

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}
