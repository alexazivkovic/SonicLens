import type { Analysis, DescriptorInfo, ModelCard, PerceptualKey } from '../types'
import { formatDuration, formatNumber, reliability } from '../format'
import { InfoTip } from './InfoTip'

interface Props {
  result: Analysis
  features: DescriptorInfo[]
  model: ModelCard | null
}

function Estimated() {
  return (
    <span className="badge" title="Estimated: a heuristic, less reliable than the other values">
      estimated
    </span>
  )
}

function PerceptualBars({ result, features, model }: Props) {
  const perceptual = features.filter((d) => d.group === 'perceptual')
  return (
    <div className="card">
      <h3>Perceptual descriptors</h3>
      <p className="muted small">
        Predicted by a model trained on FMA with Echo Nest labels. <strong>Test R²</strong> shows how well each
        descriptor is predicted on unseen artists (1 = perfect, 0 = no better than a constant).
      </p>
      <ul className="meters">
        {perceptual.map((d) => {
          const value = result.perceptual[d.key as PerceptualKey]
          const r2 = model?.targets[d.key as PerceptualKey]?.test.r2
          const rel = r2 === undefined ? null : reliability(r2)
          return (
            <li key={d.key} className="meter-row">
              <span className="meter-label">
                {d.label}
                <InfoTip label={d.label}>
                  <span>{d.description}</span>
                  {r2 !== undefined && (
                    <span className="info-metric">
                      Test R² {r2.toFixed(2)}, reliability: {rel}
                      {rel === 'low' && '. Treat this value as a rough hint only.'}
                    </span>
                  )}
                </InfoTip>
              </span>
              <span
                className="meter"
                role="meter"
                aria-label={d.label}
                aria-valuemin={0}
                aria-valuemax={1}
                aria-valuenow={Number(value.toFixed(3))}
              >
                <span className="meter-fill" style={{ width: `${value * 100}%` }} />
              </span>
              <span className="meter-value">{value.toFixed(2)}</span>
              {rel && <span className={`reliability ${rel}`}>R² {r2!.toFixed(2)}</span>}
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function CoreCards({ result }: { result: Analysis }) {
  const m = result.musical
  const cards: { label: string; value: string; unit?: string; estimated?: boolean }[] = [
    { label: 'Tempo', value: m.tempo.toFixed(1), unit: 'BPM' },
    { label: 'Key', value: `${m.key} ${m.mode}` },
    { label: 'Key confidence', value: m.key_confidence.toFixed(2) },
    { label: 'Loudness', value: m.loudness.toFixed(1), unit: 'LUFS' },
    { label: 'Time signature', value: `${m.time_signature}/4`, estimated: true },
    { label: 'Duration', value: formatDuration(m.duration_ms) },
  ]
  return (
    <div className="tiles">
      {cards.map((c) => (
        <div key={c.label} className="tile">
          <span className="tile-label">
            {c.label} {c.estimated && <Estimated />}
          </span>
          <span className="tile-value">
            {c.value}
            {c.unit && <span className="tile-unit"> {c.unit}</span>}
          </span>
        </div>
      ))}
    </div>
  )
}

function SoundCharacter({ result, features }: Props) {
  const core = features.filter((d) => d.group === 'signal' && d.tier === 'core')
  return (
    <div className="tiles">
      {core.map((d) => (
        <div key={d.key} className="tile">
          <span className="tile-label">
            {d.label}
            <InfoTip label={d.label}>{d.description}</InfoTip>
          </span>
          <span className="tile-value">
            {formatNumber(result.signal[d.key] as number)}
            {d.unit && <span className="tile-unit"> {d.unit}</span>}
          </span>
        </div>
      ))}
    </div>
  )
}

export function Overview(props: Props) {
  return (
    <section className="section" aria-labelledby="overview-title">
      <h2 id="overview-title">Overview</h2>
      <CoreCards result={props.result} />
      <PerceptualBars {...props} />
      <h3 className="subhead">Sound character</h3>
      <SoundCharacter {...props} />
    </section>
  )
}
