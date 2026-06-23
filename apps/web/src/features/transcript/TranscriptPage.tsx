import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Clock, Users, Download, Play, Pause, Loader2 } from 'lucide-react'
import { useSessionStore } from '../../store/sessionStore'
import { speakerDisplayName } from '../../lib/speakerUtils'
import type { Word } from '@pandaro/shared-types'

const SPEAKER_COLORS: Record<string, string> = {
  GŁOS_01: 'bg-blue-100 text-blue-800',
  GŁOS_02: 'bg-violet-100 text-violet-800',
  GŁOS_03: 'bg-amber-100 text-amber-800',
  GŁOS_04: 'bg-green-100 text-green-800',
  GŁOS_05: 'bg-rose-100 text-rose-800',
  GŁOS_06: 'bg-cyan-100 text-cyan-800',
}

function fmtTime(secs: number) {
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

/** Renders a single word. Low-confidence words with alternatives show a
 *  dotted amber underline; clicking opens an inline alternatives picker. */
function WordChip({
  word,
  onAccept,
}: {
  word: Word
  onAccept: (alt: string) => void
}) {
  const [open, setOpen] = useState(false)
  const hasAlts = word.alternatives && word.alternatives.length > 0

  if (!hasAlts) return <span>{word.text}</span>

  return (
    <span className="relative inline-block">
      <button
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v) }}
        className="text-amber-700 underline decoration-dotted underline-offset-2 hover:text-amber-900 transition-colors"
        title={`Pewność: ${Math.round(word.probability * 100)}% — kliknij aby zmienić`}
      >
        {word.text}
      </button>
      {open && (
        <>
          {/* invisible backdrop to catch outside clicks */}
          <span
            className="fixed inset-0 z-10"
            onClick={() => setOpen(false)}
          />
          <span className="absolute bottom-full left-0 z-20 mb-1 flex flex-col bg-white border border-amber-200 rounded-lg shadow-lg overflow-hidden min-w-max">
            <span className="text-xs text-slate-400 px-2.5 pt-1.5 pb-1 border-b border-slate-100 block">
              alternatywy
            </span>
            {word.alternatives!.map((alt, i) => (
              <button
                key={i}
                onClick={(e) => { e.stopPropagation(); onAccept(alt); setOpen(false) }}
                className="text-xs text-left px-2.5 py-1 hover:bg-amber-50 text-slate-800 whitespace-nowrap"
              >
                {alt}
              </button>
            ))}
          </span>
        </>
      )}
    </span>
  )
}

export default function TranscriptPage() {
  const { session, updateSegmentWord } = useSessionStore()
  const navigate = useNavigate()
  const [activeSegment, setActiveSegment] = useState<number | null>(null)
  const [audioSrc, setAudioSrc] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const audioRef = useRef<HTMLAudioElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const segmentRefs = useRef<Record<number, HTMLDivElement | null>>({})

  const profiles = session?.speakerProfiles ?? {}
  const spName = (sp: string) => speakerDisplayName(sp, profiles)

  // Auto-load audio from session when available (set by processing pipeline)
  useEffect(() => {
    if (session?.audioObjectUrl && !audioSrc) {
      setAudioSrc(session.audioObjectUrl)
    }
  }, [session?.audioObjectUrl, audioSrc])

  useEffect(() => {
    if (!session) navigate('/')
  }, [session, navigate])

  if (!session) return null

  const speakers = [...new Set(session.segments.map((s) => s.speaker))].sort()
  const duration = session.duration ?? session.segments[session.segments.length - 1]?.end ?? 1

  const loadAudioFile = (file: File) => {
    // Only revoke if it's a locally created URL (not the session URL)
    if (audioSrc && audioSrc !== session.audioObjectUrl) URL.revokeObjectURL(audioSrc)
    setAudioSrc(URL.createObjectURL(file))
  }

  const seekTo = useCallback((t: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = t
      audioRef.current.play().catch(() => {})
      setPlaying(true)
    }
  }, [])

  const togglePlay = () => {
    if (!audioRef.current) return
    if (playing) { audioRef.current.pause(); setPlaying(false) }
    else { audioRef.current.play().catch(() => {}); setPlaying(true) }
  }

  const exportText = () => {
    const lines = session.segments.map(
      (s) => `[${fmtTime(s.start)} - ${fmtTime(s.end)}] ${spName(s.speaker)}: ${s.text}`,
    )
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${session.fileName.replace(/\.[^.]+$/, '')}_transkrypcja.txt`
    a.click()
  }

  const SPEAKER_BAR_COLORS: Record<string, string> = {
    GŁOS_01: '#3b82f6', GŁOS_02: '#7c3aed',
    GŁOS_03: '#d97706', GŁOS_04: '#16a34a',
    GŁOS_05: '#e11d48', GŁOS_06: '#0891b2',
  }
  const DEFAULT_BAR_COLOR = '#94a3b8'

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-6 py-4 flex items-center gap-4">
        <div className="flex-1">
          <h1 className="font-semibold text-slate-900 truncate">{session.fileName}</h1>
          <div className="flex items-center gap-4 mt-1">
            {session.duration != null && (
              <span className="flex items-center gap-1 text-xs text-slate-500">
                <Clock className="w-3.5 h-3.5" />
                {fmtTime(session.duration)}
              </span>
            )}
            <span className="flex items-center gap-1 text-xs text-slate-500">
              <Users className="w-3.5 h-3.5" />
              {speakers.length} {speakers.length === 1 ? 'mówca' : 'mówców'}
            </span>
            {session.detectedLanguage && session.detectedLanguage !== 'auto' && (
              <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full">
                {session.detectedLanguage}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="text-xs px-3 py-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 transition-colors"
          >
            {audioSrc ? 'Zmień plik' : 'Załaduj audio'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,video/mp4"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && loadAudioFile(e.target.files[0])}
          />
          {audioSrc && (
            <button
              onClick={togglePlay}
              className="p-1.5 rounded-lg bg-brand-600 text-white hover:bg-brand-700 transition-colors"
            >
              {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>
          )}
          <button
            onClick={exportText}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Eksportuj
          </button>
        </div>
      </div>

      {/* Audio element */}
      {audioSrc && (
        <audio
          ref={audioRef}
          src={audioSrc}
          onEnded={() => setPlaying(false)}
          onTimeUpdate={() => {
            const t = audioRef.current?.currentTime ?? 0
            setCurrentTime(t)
            let idx = -1
            for (let i = 0; i < session.segments.length; i++) {
              if (session.segments[i].start <= t) idx = i
            }
            if (idx >= 0 && idx !== activeSegment) {
              setActiveSegment(idx)
              segmentRefs.current[idx]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
            }
          }}
          className="hidden"
        />
      )}

      {/* ── Dialog timeline bar ──────────────────────────────────────────── */}
      {session.segments.length > 0 && (
        <div className="bg-white border-b border-slate-200 px-4 pt-2 pb-1">
          {/* Speaker rows */}
          <div className="space-y-0.5">
            {speakers.map((sp) => {
              const color = SPEAKER_BAR_COLORS[sp] ?? DEFAULT_BAR_COLOR
              const segsForSp = session.segments.filter((s) => s.speaker === sp)
              return (
                <div key={sp} className="flex items-center gap-2">
                  <span
                    className="text-[10px] font-medium w-20 shrink-0 truncate text-right"
                    style={{ color }}
                  >
                    {spName(sp)}
                  </span>
                  {/* Timeline strip */}
                  <div
                    className="relative flex-1 h-3 rounded-full overflow-hidden bg-slate-100 cursor-pointer"
                    onClick={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect()
                      const pct = (e.clientX - rect.left) / rect.width
                      seekTo(pct * duration)
                    }}
                  >
                    {segsForSp.map((seg) => {
                      const left = (seg.start / duration) * 100
                      const width = Math.max(0.3, ((seg.end - seg.start) / duration) * 100)
                      const isActive = seg.start <= currentTime && currentTime < seg.end
                      return (
                        <div
                          key={seg.id}
                          className="absolute top-0 h-full rounded-sm transition-opacity"
                          style={{
                            left: `${left}%`,
                            width: `${width}%`,
                            backgroundColor: color,
                            opacity: isActive ? 1 : 0.45,
                          }}
                          onClick={(e) => { e.stopPropagation(); seekTo(seg.start) }}
                          title={`${spName(sp)} @ ${fmtTime(seg.start)}`}
                        />
                      )
                    })}
                    {/* Playhead */}
                    {audioSrc && (
                      <div
                        className="absolute top-0 h-full w-0.5 bg-slate-800 z-10 pointer-events-none"
                        style={{ left: `${(currentTime / duration) * 100}%` }}
                      />
                    )}
                  </div>
                  <span className="text-[10px] text-slate-400 w-8 shrink-0 tabular-nums">
                    {fmtTime(duration)}
                  </span>
                </div>
              )
            })}
          </div>
          {/* Time tick labels */}
          <div className="ml-[5.5rem] mr-10 flex justify-between text-[9px] text-slate-300 mt-0.5">
            {Array.from({ length: 5 }).map((_, i) => (
              <span key={i}>{fmtTime((duration / 4) * i)}</span>
            ))}
          </div>
        </div>
      )}

      {/* Speaker legend */}
      {speakers.length > 1 && (
        <div className="bg-white border-b border-slate-200 px-6 py-2 flex items-center gap-2 flex-wrap">
          {speakers.map((sp) => (
            <span
              key={sp}
              className={`text-xs px-2 py-0.5 rounded-full font-medium ${SPEAKER_COLORS[sp] ?? 'bg-slate-100 text-slate-600'}`}
            >
              {spName(sp)}
              {spName(sp) !== sp && (
                <span className="ml-1 opacity-50">({sp})</span>
              )}
            </span>
          ))}
        </div>
      )}

      {/* Segments */}
      <div className="flex-1 overflow-auto px-6 py-4 space-y-1">
        {session.processing.step !== 'done' && session.processing.step !== 'error' && (
          <div className="flex items-center gap-2 mb-3 px-3 py-2 bg-brand-50 border border-brand-200 rounded-xl text-sm text-brand-700">
            <Loader2 className="w-4 h-4 animate-spin shrink-0" />
            <span>{session.processing.message || 'Przetwarzanie…'}</span>
            <span className="ml-auto text-xs text-brand-500">{session.processing.progress}%</span>
          </div>
        )}
        {session.segments.length === 0 && session.processing.step === 'done' && (
          <p className="text-slate-400 text-center py-12">Brak segmentów transkrypcji.</p>
        )}
        {session.segments.length === 0 && session.processing.step !== 'done' && session.processing.step !== 'error' && (
          <p className="text-slate-400 text-center py-12">Segmenty pojawią się tutaj podczas przetwarzania…</p>
        )}
        {session.segments.map((seg, idx) => {
          const isActive = idx === activeSegment
          const hasTranslation = seg.text_pl && seg.text_pl !== seg.text
          const hasWords = seg.words && seg.words.length > 0
          const hasAnyAlts = hasWords && seg.words!.some((w) => w.alternatives && w.alternatives.length > 0)
          return (
            <div
              key={seg.id}
              ref={(el) => { segmentRefs.current[idx] = el }}
              className={[
                'flex gap-3 p-3 rounded-xl transition-colors cursor-pointer',
                isActive ? 'bg-brand-50 ring-1 ring-brand-200' : 'hover:bg-slate-50',
              ].join(' ')}
              onClick={() => seekTo(seg.start)}
            >
              <span className="text-xs text-slate-400 w-20 shrink-0 pt-0.5 tabular-nums">
                {fmtTime(seg.start)}
              </span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className={`text-xs font-semibold ${SPEAKER_COLORS[seg.speaker] ?? 'text-slate-600'}`}>
                    {spName(seg.speaker)}
                  </span>
                  {hasAnyAlts && (
                    <span className="text-xs text-amber-600 font-medium" title="Segment zawiera słowa z niską pewnością">
                      ✱ niepewne słowa
                    </span>
                  )}
                </div>

                {hasTranslation && (
                  <span className="text-slate-800 text-sm leading-relaxed block">
                    {seg.text_pl}
                  </span>
                )}

                <span
                  className={[
                    'text-sm leading-relaxed',
                    hasTranslation ? 'text-slate-400 text-xs italic' : 'text-slate-800',
                  ].join(' ')}
                >
                  {hasTranslation && '('}
                  {hasWords
                    ? seg.words!.map((w, wi) => (
                        <WordChip
                          key={wi}
                          word={w}
                          onAccept={(alt) => updateSegmentWord(seg.id, wi, alt)}
                        />
                      ))
                    : seg.text
                  }
                  {hasTranslation && ')'}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
