import { useId, useState } from 'react'
import type { ReactNode } from 'react'

/**
 * A small "i" button that reveals an explanation on hover, keyboard focus or click/tap.
 * A click always opens it (never toggles, which would fight the hover and focus that precede
 * the click); it closes on mouse leave, blur or Escape.
 */
export function InfoTip({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const id = useId()
  return (
    <span className="info" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button
        type="button"
        className="info-button"
        aria-label={`About ${label}`}
        aria-expanded={open}
        aria-describedby={open ? id : undefined}
        onClick={() => setOpen(true)}
        onFocus={(e) => e.currentTarget.matches(':focus-visible') && setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => e.key === 'Escape' && setOpen(false)}
      >
        i
      </button>
      {open && (
        <span className="info-pop" role="tooltip" id={id}>
          {children}
        </span>
      )}
    </span>
  )
}
