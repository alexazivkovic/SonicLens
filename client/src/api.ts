import type { Analysis, DescriptorInfo, Health, ModelCard } from './types'

export const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

// Mirrors the server's defaults; the server remains the authority and returns 413/415 itself.
export const MAX_UPLOAD_MB = 50
export const ACCEPTED_EXTENSIONS = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_URL}${path}`, init)
  } catch {
    throw new ApiError(`Cannot reach the SonicLens API at ${API_URL}. Is the backend running?`)
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join('; ')
    } catch {
      /* keep the status line */
    }
    throw new ApiError(detail)
  }
  return res.json() as Promise<T>
}

export const getHealth = () => request<Health>('/health')
export const getFeatures = () => request<DescriptorInfo[]>('/features')
export const getModelCard = () => request<ModelCard>('/model')

export function analyze(file: File, signal?: AbortSignal): Promise<Analysis> {
  const body = new FormData()
  body.append('file', file)
  return request<Analysis>('/analyze', { method: 'POST', body, signal })
}

/** Client-side pre-check so obvious mistakes fail instantly; returns an error message or null. */
export function checkFile(file: File): string | null {
  const dot = file.name.lastIndexOf('.')
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : ''
  if (!ACCEPTED_EXTENSIONS.includes(ext)) {
    return `"${file.name}" is not a supported format. Use ${ACCEPTED_EXTENSIONS.join(', ')}.`
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return `"${file.name}" is ${(file.size / 1024 / 1024).toFixed(0)} MB; the limit is ${MAX_UPLOAD_MB} MB.`
  }
  return null
}
