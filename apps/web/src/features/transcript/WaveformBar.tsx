/**
 * WaveformBar — audio waveform canvas rendered from the decoded PCM data.
 *
 * Features:
 * - Downsampled peaks visualisation (fast, works on multi-hour audio)
 * - Noise / silence region overlay (orange = noise, grey = silence)
 * - Live playhead that moves with audio.currentTime
 * - Click-to-seek: clicking anywhere on the waveform seeks the audio
 */
import { useEffect, useRef, useCallback } from 'react'

interface NoiseRegion {
  start_sec: number
  end_sec: number
  type: 'silence' | 'noise'
}

interface Selection {
  start: number
  end: number
}

interface Props {
  audioSrc: string | null
  duration: number
  currentTime: number
  noiseRegions?: NoiseRegion[]
  onSeek: (t: number) => void
  height?: number
  selection?: Selection | null
  onSelectionChange?: (sel: Selection | null) => void
}

const WAVEFORM_PEAKS_PER_PX = 2   // samples per pixel column (increases resolution)
const WAVEFORM_TARGET_COLUMNS = 1200  // max columns decoded

export function WaveformBar({
  audioSrc,
  duration,
  currentTime,
  noiseRegions = [],
  onSeek,
  height = 48,
  selection = null,
  onSelectionChange,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const peaksRef  = useRef<Float32Array | null>(null)
  const rafRef    = useRef<number>(0)
  const dragRef   = useRef<{ anchor: number; moved: boolean } | null>(null)

  // ── Decode audio + compute peaks ──────────────────────────────────────────
  useEffect(() => {
    if (!audioSrc) return
    let cancelled = false

    const ctx = new AudioContext()
    fetch(audioSrc)
      .then(r => r.arrayBuffer())
      .then(buf => ctx.decodeAudioData(buf))
      .then(decoded => {
        if (cancelled) return
        const ch = decoded.getChannelData(0)
        const totalSamples = ch.length
        const columns = Math.min(WAVEFORM_TARGET_COLUMNS, totalSamples)
        const blockSize = Math.floor(totalSamples / columns)
        const peaks = new Float32Array(columns)
        for (let i = 0; i < columns; i++) {
          let max = 0
          const start = i * blockSize
          for (let j = 0; j < blockSize; j++) {
            const v = Math.abs(ch[start + j] ?? 0)
            if (v > max) max = v
          }
          peaks[i] = max
        }
        peaksRef.current = peaks
        drawFrame()
      })
      .catch(() => {/* non-fatal */})
      .finally(() => ctx.close())

    return () => { cancelled = true }
  }, [audioSrc])  // eslint-disable-line react-hooks/exhaustive-deps

  // ── Draw frame ────────────────────────────────────────────────────────────
  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current
    const peaks  = peaksRef.current
    if (!canvas || !peaks) return

    const W = canvas.width
    const H = canvas.height
    const dpr = window.devicePixelRatio || 1
    const w   = W / dpr
    const midY = H / (2 * dpr)

    const ctx2d = canvas.getContext('2d')!
    ctx2d.clearRect(0, 0, W, H)

    const secToX = (sec: number) => (sec / Math.max(duration, 1)) * w * dpr

    // ── Noise / silence regions ──────────────────────────────────────────────
    for (const region of noiseRegions) {
      const x1 = secToX(region.start_sec)
      const x2 = secToX(region.end_sec)
      ctx2d.fillStyle = region.type === 'silence'
        ? 'rgba(148,163,184,0.25)'   // slate-300 / 25%
        : 'rgba(251,191,36,0.20)'    // amber-400 / 20%
      ctx2d.fillRect(x1, 0, Math.max(x2 - x1, 1), H)
    }

    // ── Time-range selection overlay ──────────────────────────────────────────
    if (selection && selection.end > selection.start) {
      const sx1 = secToX(selection.start)
      const sx2 = secToX(selection.end)
      ctx2d.fillStyle = 'rgba(99,102,241,0.18)'   // indigo-500 / 18%
      ctx2d.fillRect(sx1, 0, Math.max(sx2 - sx1, 1), H)
      ctx2d.strokeStyle = 'rgba(79,70,229,0.7)'
      ctx2d.lineWidth = 1 * dpr
      ctx2d.strokeRect(sx1, 0, Math.max(sx2 - sx1, 1), H)
    }

    // ── Waveform bars ────────────────────────────────────────────────────────
    const playheadX = secToX(currentTime)
    const nPeaks = peaks.length

    for (let i = 0; i < nPeaks; i++) {
      const x = (i / nPeaks) * w * dpr
      const barH = peaks[i] * H * 0.85
      const isPast = x <= playheadX
      ctx2d.fillStyle = isPast ? '#6366f1' : '#cbd5e1'  // indigo vs slate-300
      ctx2d.fillRect(
        Math.round(x),
        Math.round(midY * dpr - barH / 2),
        Math.max(1, Math.round((w * dpr) / nPeaks) - 1),
        Math.round(barH),
      )
    }

    // ── Playhead ─────────────────────────────────────────────────────────────
    if (duration > 0) {
      ctx2d.strokeStyle = '#4f46e5'
      ctx2d.lineWidth   = 2 * dpr
      ctx2d.beginPath()
      ctx2d.moveTo(playheadX, 0)
      ctx2d.lineTo(playheadX, H)
      ctx2d.stroke()
    }
  }, [currentTime, duration, noiseRegions, selection])

  // Redraw when currentTime changes
  useEffect(() => {
    cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(drawFrame)
    return () => cancelAnimationFrame(rafRef.current)
  }, [drawFrame])

  // Resize observer so the canvas fills its container
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const obs = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1
      canvas.width  = canvas.offsetWidth  * dpr
      canvas.height = canvas.offsetHeight * dpr
      drawFrame()
    })
    obs.observe(canvas)
    return () => obs.disconnect()
  }, [drawFrame])

  const xToSec = useCallback((clientX: number, el: HTMLCanvasElement) => {
    const rect = el.getBoundingClientRect()
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    return pct * duration
  }, [duration])

  const handlePointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!onSelectionChange) {
      onSeek(xToSec(e.clientX, e.currentTarget))
      return
    }
    e.currentTarget.setPointerCapture(e.pointerId)
    dragRef.current = { anchor: xToSec(e.clientX, e.currentTarget), moved: false }
  }, [onSelectionChange, onSeek, xToSec])

  const handlePointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current
    if (!drag || !onSelectionChange) return
    const cur = xToSec(e.clientX, e.currentTarget)
    if (Math.abs(cur - drag.anchor) > 0.05) drag.moved = true
    if (drag.moved) {
      onSelectionChange({ start: Math.min(drag.anchor, cur), end: Math.max(drag.anchor, cur) })
    }
  }, [onSelectionChange, xToSec])

  const handlePointerUp = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current
    dragRef.current = null
    if (!drag) return
    if (!drag.moved) {
      // Treat as a plain click: clear selection and seek
      onSelectionChange?.(null)
      onSeek(xToSec(e.clientX, e.currentTarget))
    }
  }, [onSelectionChange, onSeek, xToSec])

  if (!audioSrc) return null

  return (
    <div style={{ height }} className="w-full relative">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-pointer rounded select-none"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        title={onSelectionChange ? 'Kliknij, aby przewinąć • przeciągnij, aby zaznaczyć fragment' : 'Kliknij, aby przewinąć'}
      />
      {/* Legend for noise regions */}
      {noiseRegions.length > 0 && (
        <div className="absolute top-1 right-2 flex items-center gap-2 text-[9px] text-slate-400 pointer-events-none">
          <span className="flex items-center gap-0.5">
            <span className="w-2 h-2 rounded-sm bg-slate-300/60 inline-block" />
            cisza
          </span>
        </div>
      )}
    </div>
  )
}
