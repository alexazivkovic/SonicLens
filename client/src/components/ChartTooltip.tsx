import { useCallback, useState } from 'react'
import type { FocusEvent, PointerEvent } from 'react'

export interface TipContent {
  value: string
  label: string
}

interface TipState extends TipContent {
  x: number
  y: number
}

/**
 * One tooltip per chart. Marks spread `bind(content)` to get hover and keyboard-focus handlers;
 * the value is shown first (strong), the label second. Coordinates are relative to the chart
 * container, which must be `position: relative`.
 */
export function useChartTooltip() {
  const [tip, setTip] = useState<TipState | null>(null)

  const place = useCallback((el: Element, content: TipContent, clientX?: number) => {
    const box = el.closest('.chart')?.getBoundingClientRect()
    // Anchor to the visible mark when the hit area is a larger invisible rect in front of it.
    const next = el.nextElementSibling
    const mark = (next?.classList.contains('bar') ? next : el).getBoundingClientRect()
    if (!box) return
    const x = (clientX ?? mark.left + mark.width / 2) - box.left
    setTip({ ...content, x, y: mark.top - box.top })
  }, [])

  const bind = useCallback(
    (content: TipContent) => ({
      tabIndex: 0,
      'aria-label': `${content.label}: ${content.value}`,
      onPointerMove: (e: PointerEvent<Element>) => place(e.currentTarget, content, e.clientX),
      onPointerLeave: () => setTip(null),
      onFocus: (e: FocusEvent<Element>) => place(e.currentTarget, content),
      onBlur: () => setTip(null),
    }),
    [place],
  )

  const element = tip ? (
    <div className="chart-tip" style={{ left: tip.x, top: tip.y }} role="tooltip">
      <strong>{tip.value}</strong>
      <span>{tip.label}</span>
    </div>
  ) : null

  return { bind, element }
}
