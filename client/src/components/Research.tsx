import { useEffect, useRef, useState } from 'react'
import type { Analysis } from '../types'
import { PITCH_CLASSES, formatNumber, formatSeconds } from '../format'
import { useChartTooltip } from './ChartTooltip'

const DETAIL_SPAN_S = 15

/** Width of an element in CSS pixels, kept up to date on resize. */
function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(600)
  useEffect(() => {
    if (!ref.current) return
    const obs = new ResizeObserver(([entry]) => setWidth(Math.max(240, entry.contentRect.width)))
    obs.observe(ref.current)
    return () => obs.disconnect()
  }, [])
  return [ref, width] as const
}

function niceTicks(min: number, max: number, count = 4, multipliers = [1, 2, 2.5, 5, 10]): number[] {
  const span = max - min || 1
  const raw = span / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = multipliers.map((m) => m * mag).find((s) => span / s <= count) ?? raw
  const ticks = []
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(Number(v.toFixed(10)))
  return ticks
}

/** Clean tick labels: 0.25, 0.5, 1, 75 (no padded decimals). */
const formatTick = (t: number) => String(Number(t.toFixed(6)))

/** A bar from the baseline with a 4px rounded data end (top for positive, bottom for negative). */
function barPath(x: number, w: number, y0: number, y1: number): string {
  const r = Math.min(4, w / 2, Math.abs(y1 - y0))
  if (y1 <= y0) {
    return `M${x},${y0} V${y1 + r} Q${x},${y1} ${x + r},${y1} H${x + w - r} Q${x + w},${y1} ${x + w},${y1 + r} V${y0} Z`
  }
  return `M${x},${y0} V${y1 - r} Q${x},${y1} ${x + r},${y1} H${x + w - r} Q${x + w},${y1} ${x + w},${y1 - r} V${y0} Z`
}

interface ColumnChartProps {
  title: string
  labels: string[]
  values: number[]
  domain?: [number, number]
  tipLabel: (i: number) => string
  directLabel?: number // index of the one bar that gets a value label
}

function ColumnChart({ title, labels, values, domain, tipLabel, directLabel }: ColumnChartProps) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const { bind, element } = useChartTooltip()
  const height = 200
  const m = { top: 18, right: 8, bottom: 26, left: 40 }
  const lo = domain?.[0] ?? Math.min(0, ...values)
  const hi = domain?.[1] ?? Math.max(0, ...values)
  const ticks = niceTicks(lo, hi)
  const yMin = Math.min(lo, ticks[0]), yMax = Math.max(hi, ticks[ticks.length - 1])
  const plotW = width - m.left - m.right
  const plotH = height - m.top - m.bottom
  const y = (v: number) => m.top + plotH * (1 - (v - yMin) / (yMax - yMin))
  const band = plotW / values.length
  const barW = Math.min(24, band * 0.62)

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label={title}>
        {ticks.map((t) => (
          <g key={t}>
            <line className={t === 0 ? 'axis' : 'grid'} x1={m.left} x2={width - m.right} y1={y(t)} y2={y(t)} />
            <text className="tick" x={m.left - 6} y={y(t)} textAnchor="end" dominantBaseline="middle">
              {formatTick(t)}
            </text>
          </g>
        ))}
        {values.map((v, i) => {
          const x = m.left + band * i + (band - barW) / 2
          return (
            <g key={i}>
              <rect className="hit" x={m.left + band * i} y={m.top} width={band} height={plotH} {...bind({ value: formatNumber(v), label: tipLabel(i) })} />
              <path className="bar" d={barPath(x, barW, y(0), y(v))} pointerEvents="none" />
              {directLabel === i && (
                <text className="value-label" x={x + barW / 2} y={v >= 0 ? y(v) - 5 : y(v) + 12} textAnchor="middle">
                  {formatNumber(v)}
                </text>
              )}
              <text className="tick" x={m.left + band * i + band / 2} y={height - 8} textAnchor="middle">
                {labels[i]}
              </text>
            </g>
          )
        })}
      </svg>
      {element}
    </div>
  )
}

function BeatTimeline({ result }: { result: Analysis }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const { bind, element } = useChartTooltip()
  const duration = result.musical.duration_ms / 1000
  const { beat_grid: beats, downbeats } = result.research
  const win = result.model_input.window
  const span = Math.min(DETAIL_SPAN_S, duration)
  const [start, setStart] = useState(() => Math.max(0, Math.min(win.start_s, duration - span)))
  const end = start + span

  const m = { left: 8, right: 8 }
  const plotW = width - m.left - m.right
  const xo = (t: number) => m.left + (t / duration) * plotW
  const xd = (t: number) => m.left + ((t - start) / span) * plotW
  const downSet = new Set(downbeats.map((t) => t.toFixed(4)))
  const visible = beats.map((t, i) => ({ t, i })).filter(({ t }) => t >= start && t <= end)
  const axisTicks = niceTicks(start, end, width < 500 ? 3 : 6, [1, 2, 5, 10]) // whole seconds on a time axis.filter((t) => t >= start && t <= end)

  const overviewH = 40
  const detailTop = overviewH + 18
  const detailH = 70
  const height = detailTop + detailH + 22

  return (
    <div className="chart" ref={ref}>
      <ul className="legend" aria-label="Legend">
        <li><span className="key key-beat" />Beats</li>
        <li><span className="key key-downbeat" />Downbeats <span className="badge">estimated</span></li>
        <li><span className="key key-window" />Model window ({formatSeconds(win.start_s)} to {formatSeconds(win.end_s)})</li>
      </ul>
      <svg width={width} height={height} role="img" aria-label="Beat grid and downbeats over time">
        {/* Overview: whole track */}
        <rect className="window-band" x={xo(win.start_s)} y={0} width={xo(win.end_s) - xo(win.start_s)} height={overviewH} />
        <line className="axis" x1={m.left} x2={width - m.right} y1={overviewH} y2={overviewH} />
        {downbeats.map((t) => (
          <line key={t} className="downbeat thin" x1={xo(t)} x2={xo(t)} y1={overviewH * 0.35} y2={overviewH} />
        ))}
        <rect className="viewport" x={xo(start)} y={1} width={Math.max(2, xo(end) - xo(start))} height={overviewH - 2} rx={3} />
        <rect
          className="hit"
          x={m.left}
          y={0}
          width={plotW}
          height={overviewH}
          style={{ cursor: 'pointer' }}
          onClick={(e) => {
            const box = e.currentTarget.getBoundingClientRect()
            const t = ((e.clientX - box.left) / box.width) * duration
            setStart(Math.max(0, Math.min(duration - span, t - span / 2)))
          }}
        />
        <text className="tick" x={m.left} y={overviewH + 13}>0:00</text>
        <text className="tick" x={width - m.right} y={overviewH + 13} textAnchor="end">
          {formatSeconds(duration)}
        </text>

        {/* Detail: the selected span */}
        <g transform={`translate(0, ${detailTop})`}>
          {win.end_s > start && win.start_s < end && (
            <rect className="window-band" x={xd(Math.max(start, win.start_s))} y={0}
              width={xd(Math.min(end, win.end_s)) - xd(Math.max(start, win.start_s))} height={detailH} />
          )}
          <line className="axis" x1={m.left} x2={width - m.right} y1={detailH} y2={detailH} />
          {axisTicks.map((t) => (
            <text key={t} className="tick" x={xd(t)} y={detailH + 15} textAnchor="middle">
              {t < 60 ? `${formatTick(t)} s` : `${Math.floor(t / 60)}:${formatTick(t % 60).padStart(2, '0')}`}
            </text>
          ))}
          {visible.map(({ t, i }) => {
            const isDown = downSet.has(t.toFixed(4))
            const bar = isDown ? downbeats.findIndex((d) => d.toFixed(4) === t.toFixed(4)) + 1 : 0
            return (
              <g key={i}>
                <line className={isDown ? 'downbeat' : 'beat'} x1={xd(t)} x2={xd(t)} y1={isDown ? 4 : detailH * 0.4} y2={detailH} />
                <rect className="hit" x={xd(t) - 6} y={0} width={12} height={detailH}
                  {...bind({ value: formatSeconds(t), label: isDown ? `Downbeat, bar ${bar} (beat ${i + 1})` : `Beat ${i + 1}` })} />
              </g>
            )
          })}
        </g>
      </svg>
      {duration > span && (
        <label className="slider">
          <span className="muted small">
            Detail view: {formatSeconds(start)} to {formatSeconds(end)} (click the overview or use the slider)
          </span>
          <input type="range" min={0} max={duration - span} step={0.25} value={start}
            onChange={(e) => setStart(Number(e.target.value))} />
        </label>
      )}
      {element}
    </div>
  )
}

export function ResearchSection({ result }: { result: Analysis }) {
  const chroma = result.research.chroma_vector
  const peak = chroma.indexOf(Math.max(...chroma))
  const mfcc = result.research.mfcc
  return (
    <section className="section" aria-labelledby="research-title">
      <h2 id="research-title">Research</h2>
      <div className="grid-2">
        <div className="card">
          <h3>Chroma</h3>
          <p className="muted small">Mean harmonic pitch-class profile over the track (strongest pitch class = 1).</p>
          <ColumnChart title="Chroma vector" labels={PITCH_CLASSES} values={chroma} domain={[0, 1]}
            tipLabel={(i) => `Pitch class ${PITCH_CLASSES[i]}`} directLabel={peak} />
        </div>
        <div className="card">
          <h3>MFCC (mean)</h3>
          <p className="muted small">
            Coefficients 1-12, a compact description of timbre. Coefficient 0 (overall log-energy) is{' '}
            {formatNumber(mfcc.mean[0])} and is left out so it does not dwarf the others.
          </p>
          <ColumnChart title="MFCC means 1 to 12" labels={mfcc.mean.slice(1).map((_, i) => String(i + 1))}
            values={mfcc.mean.slice(1)} tipLabel={(i) => `MFCC ${i + 1}, std ${formatNumber(mfcc.std[i + 1])}`} />
        </div>
      </div>
      <div className="card">
        <h3>Beat grid</h3>
        <p className="muted small">
          {result.research.beat_grid.length} beats, {result.research.downbeats.length} estimated downbeats. The shaded
          span is the 30 s window the perceptual descriptors were computed on.
        </p>
        <BeatTimeline result={result} />
      </div>
    </section>
  )
}
