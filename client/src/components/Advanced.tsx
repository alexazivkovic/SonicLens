import type { Analysis, DescriptorInfo, Family } from '../types'
import { formatValue } from '../format'

const FAMILY_TITLES: Record<Family, string> = {
  spectral: 'Spectral',
  timbre: 'Timbre and noise',
  dynamics: 'Dynamics',
  rhythm: 'Rhythm',
  tonal: 'Tonal',
}

export function AdvancedSection({ result, features }: { result: Analysis; features: DescriptorInfo[] }) {
  const signal = features.filter((d) => d.group === 'signal')
  const families = Object.keys(FAMILY_TITLES) as Family[]
  return (
    <section className="section">
      <details className="card advanced">
        <summary>
          <h2>Advanced: all signal descriptors</h2>
          <span className="muted small">{signal.length} DSP descriptors, computed on the full track</span>
        </summary>
        {families.map((family) => {
          const rows = signal.filter((d) => d.family === family)
          if (!rows.length) return null
          return (
            <table key={family} className="descriptor-table">
              <caption>{FAMILY_TITLES[family]}</caption>
              <thead>
                <tr>
                  <th scope="col">Descriptor</th>
                  <th scope="col" className="num">Value</th>
                  <th scope="col">Unit</th>
                  <th scope="col">Description</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((d) => (
                  <tr key={d.key}>
                    <th scope="row">
                      {d.label}
                      {d.tier === 'core' && <span className="badge core">core</span>}
                      <code>{d.key}</code>
                    </th>
                    <td className="num">{formatValue(result.signal[d.key])}</td>
                    <td>{d.unit ?? ''}</td>
                    <td className="muted">
                      {d.description} <span className="algo">Essentia: {d.algorithm}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )
        })}
      </details>
    </section>
  )
}
