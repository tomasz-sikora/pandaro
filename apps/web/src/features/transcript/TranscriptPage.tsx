import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Clock, Users, Download, Play, Pause, Loader2, Languages, Radio, Pencil, Check, X, RefreshCw, Mic2 } from 'lucide-react'
import { useSessionStore } from '../../store/sessionStore'
import { useSettingsStore } from '../../store/settingsStore'
import { speakerDisplayName } from '../../lib/speakerUtils'
import { useAgentPipeline } from '../../hooks/useAgentPipeline'
import { WaveformBar } from './WaveformBar'
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

/** Inline editable speaker label pill. Double-click or click the pencil to edit. */
function SpeakerEditPill({
  speakerId,
  displayName,
  colorClass,
  gender,
  onRename,
}: {
  speakerId: string
  displayName: string
  colorClass: string
  gender?: string | null
  onRename: (newName: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(displayName)
  const inputRef = useRef<HTMLInputElement>(null)

  const startEdit = () => {
    setDraft(displayName)
    setEditing(true)
    setTimeout(() => inputRef.current?.select(), 0)
  }

  const commit = () => {
    const trimmed = draft.trim()
    if (trimmed && trimmed !== displayName) onRename(trimmed)
    setEditing(false)
  }

  const GENDER_ICON: Record<string, string> = { meski: '♂', zenski: '♀', dziecko: '🧒' }

  if (editing) {
    return (
      <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full font-medium ${colorClass}`}>
        <input
          ref={inputRef}
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') setEditing(false) }}
          onBlur={commit}
          className="bg-transparent outline-none w-24 text-xs"
          autoFocus
        />
        <button onClick={commit} className="opacity-70 hover:opacity-100"><Check className="w-3 h-3" /></button>
        <button onClick={() => setEditing(false)} className="opacity-70 hover:opacity-100"><X className="w-3 h-3" /></button>
      </span>
    )
  }

  return (
    <span
      className={`group inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full font-medium cursor-pointer ${colorClass}`}
      onDoubleClick={startEdit}
      title="Kliknij dwukrotnie aby zmienić nazwę mówcy"
    >
      {gender && GENDER_ICON[gender] && <span className="opacity-60 text-[10px]">{GENDER_ICON[gender]}</span>}
      {displayName}
      <button
        onClick={e => { e.stopPropagation(); startEdit() }}
        className="opacity-0 group-hover:opacity-50 hover:!opacity-100 transition-opacity ml-0.5"
        title="Zmień nazwę"
      >
        <Pencil className="w-2.5 h-2.5" />
      </button>
    </span>
  )
}

/** Non-word sound badge (mhm, cough, etc.) */
function SoundBadge({ text }: { text: string }) {
  const NON_WORD_SOUNDS: Record<string, string> = {
    mhm: '👍 mhm', hmm: '🤔 hmm', hm: '🤔 hm', aha: '💡 aha',
    ugh: '😩 ugh', uhh: '🔇 uhh', uh: '🔇 uh', eh: '🤷 eh',
    '[cough]': '🤧 kaszel', '[laugh]': '😄 śmiech', '[noise]': '🔊 szum',
    '[music]': '🎵 muzyka', '[applause]': '👏 aplauz',
  }
  const clean = text.toLowerCase().replace(/[.,!?]/g, '').trim()
  const label = NON_WORD_SOUNDS[clean]
  if (!label) return <span>{text}</span>
  return (
    <span className="inline-flex items-center gap-0.5 text-xs bg-slate-100 text-slate-500 rounded px-1 mx-0.5 font-mono italic">
      {label}
    </span>
  )
}

export default function TranscriptPage() {
  const { session, updateSegmentWord, setSpeakerDisplayName } = useSessionStore()
  const { settings } = useSettingsStore()
  const navigate = useNavigate()
  const [activeSegment, setActiveSegment] = useState<number | null>(null)
  const [audioSrc, setAudioSrc] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [showTranslation, setShowTranslation] = useState(true)
  const [selection, setSelection] = useState<{ start: number; end: number } | null>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const segmentRefs = useRef<Record<number, HTMLDivElement | null>>({})
  const { reprocessFragment } = useAgentPipeline()

  const profiles = session?.speakerProfiles ?? {}
  const spName = (sp: string) => speakerDisplayName(sp, profiles)
  const segmentQuality = session?.segmentQuality ?? {}

  // Confidence → background colour (0=red, 0.5=amber, 0.75+=transparent)
  const confidenceColor = (idx: number): string => {
    const conf = segmentQuality[idx]
    if (conf === undefined) return ''
    if (conf < 0.5) return 'bg-red-50 border-l-2 border-red-300'
    if (conf < 0.70) return 'bg-amber-50 border-l-2 border-amber-300'
    return ''
  }

  const isProcessing =
    session !== null &&
    session.processing.step !== 'idle' &&
    session.processing.step !== 'done' &&
    session.processing.step !== 'error'

  const isNonPolish = !!(session?.detectedLanguage && session.detectedLanguage !== 'pl' && session.detectedLanguage !== 'auto')
  const translatedCount = session?.segments.filter(s => s.text_pl && s.text_pl !== s.text).length ?? 0
  const hasAnyTranslation = translatedCount > 0

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
    const lines = session.segments.map((s) => {
      const header = `[${fmtTime(s.start)} - ${fmtTime(s.end)}] ${spName(s.speaker)}:`
      const hasTranslation = s.text_pl && s.text_pl !== s.text
      if (hasTranslation) return `${header} ${s.text_pl}\n  (${s.text})`
      return `${header} ${s.text}`
    })
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
      {/* Live processing banner */}
      {isProcessing && (
        <div className="bg-brand-600 text-white px-4 py-2 flex items-center gap-3 text-sm shrink-0">
          <Radio className="w-4 h-4 animate-pulse shrink-0" />
          <span className="font-medium">Agent przetwarza na żywo</span>
          <span className="text-brand-200 text-xs">{session?.processing.message}</span>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-brand-200 text-xs">{session?.segments.length} segm.</span>
            <div className="w-24 bg-brand-500 rounded-full h-1">
              <div
                className="bg-white h-1 rounded-full transition-all duration-300"
                style={{ width: `${session?.processing.progress ?? 0}%` }}
              />
            </div>
          </div>
        </div>
      )}

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
            {isNonPolish && hasAnyTranslation && (
              <span className="text-xs text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">
                {translatedCount}/{session.segments.length} → PL
              </span>
            )}
            {isNonPolish && !hasAnyTranslation && session.processing.step === 'done' && (
              <span className="text-xs text-amber-600 bg-amber-50 px-2 py-0.5 rounded-full">
                brak tłumaczenia
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
          {isNonPolish && hasAnyTranslation && (
            <button
              onClick={() => setShowTranslation((v) => !v)}
              title={showTranslation ? 'Ukryj tłumaczenie' : 'Pokaż tłumaczenie na polski'}
              className={[
                'flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors',
                showTranslation
                  ? 'border-emerald-300 bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
                  : 'border-slate-200 text-slate-500 hover:bg-slate-50',
              ].join(' ')}
            >
              <Languages className="w-3.5 h-3.5" />
              PL
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
          {/* Waveform — shown when audio is loaded */}
          {audioSrc && (
            <div className="ml-[5.5rem] mr-10 mt-1">
              <WaveformBar
                audioSrc={audioSrc}
                duration={duration}
                currentTime={currentTime}
                noiseRegions={(session as any).noiseRegions ?? []}
                onSeek={seekTo}
                height={40}
                selection={selection}
                onSelectionChange={setSelection}
              />
              {/* Fragment re-process toolbar — appears when a range is selected */}
              {selection && selection.end > selection.start && (
                <div className="mt-1.5 flex items-center gap-2 flex-wrap text-xs bg-indigo-50 border border-indigo-100 rounded-lg px-2.5 py-1.5">
                  <span className="text-indigo-700 font-medium">
                    Zaznaczono {fmtTime(selection.start)}–{fmtTime(selection.end)}:
                  </span>
                  <button
                    onClick={() => { reprocessFragment(selection.start, selection.end, 'transcription'); setSelection(null) }}
                    disabled={isProcessing || !session.sourceFile}
                    className="flex items-center gap-1 px-2 py-0.5 rounded bg-white border border-indigo-200 text-indigo-700 hover:bg-indigo-100 disabled:opacity-40 disabled:cursor-not-allowed"
                    title="Ponów transkrypcję tego fragmentu"
                  >
                    <RefreshCw className="w-3 h-3" /> Transkrypcja
                  </button>
                  <button
                    onClick={() => { reprocessFragment(selection.start, selection.end, 'diarization'); setSelection(null) }}
                    disabled={isProcessing || !session.sourceFile}
                    className="flex items-center gap-1 px-2 py-0.5 rounded bg-white border border-indigo-200 text-indigo-700 hover:bg-indigo-100 disabled:opacity-40 disabled:cursor-not-allowed"
                    title="Ponów diaryzację (przypisanie mówców) tego fragmentu"
                  >
                    <Users className="w-3 h-3" /> Diaryzacja
                  </button>
                  <button
                    onClick={() => { reprocessFragment(selection.start, selection.end, 'translation'); setSelection(null) }}
                    disabled={isProcessing || !session.sourceFile}
                    className="flex items-center gap-1 px-2 py-0.5 rounded bg-white border border-indigo-200 text-indigo-700 hover:bg-indigo-100 disabled:opacity-40 disabled:cursor-not-allowed"
                    title="Ponów tłumaczenie tego fragmentu"
                  >
                    <Languages className="w-3 h-3" /> Tłumaczenie
                  </button>
                  <button
                    onClick={() => setSelection(null)}
                    className="ml-auto text-slate-400 hover:text-slate-600"
                    title="Anuluj zaznaczenie"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Speaker legend with inline editing */}
      {speakers.length > 1 && (
        <div className="bg-white border-b border-slate-200 px-4 py-2 flex items-center gap-2 flex-wrap">
          <Mic2 className="w-3.5 h-3.5 text-slate-400 shrink-0" />
          {speakers.map((sp) => {
            const profile = profiles[sp]
            return (
              <SpeakerEditPill
                key={sp}
                speakerId={sp}
                displayName={spName(sp)}
                colorClass={SPEAKER_COLORS[sp] ?? 'bg-slate-100 text-slate-600'}
                gender={profile?.gender ?? null}
                onRename={(name) => setSpeakerDisplayName?.(sp, name)}
              />
            )
          })}
          <span className="text-[10px] text-slate-300 ml-1 shrink-0">dwuklik = zmień nazwę</span>
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
          const hasTranslation = !!(seg.text_pl && seg.text_pl !== seg.text)
          const translationMissing = isNonPolish && !hasTranslation
          const hasWords = seg.words && seg.words.length > 0
          const hasAnyAlts = hasWords && seg.words!.some((w) => w.alternatives && w.alternatives.length > 0)
          const segTags: string[] = (seg as any).tags ?? []
          const isInterjection = segTags.includes('interjection')
          const isLowConf = segTags.includes('low-conf')
          const hasIPA = segTags.includes('ipa')
          const isOverlapping = seg.overlapping === true
          const profile = profiles[seg.speaker]
          return (
            <div
              key={seg.id}
              ref={(el) => { segmentRefs.current[idx] = el }}
              className={[
                'group flex gap-3 p-3 rounded-xl transition-colors cursor-pointer relative',
                isActive ? 'bg-brand-50 ring-1 ring-brand-200' : `hover:bg-slate-50 ${confidenceColor(idx)}`,
              ].join(' ')}
              onClick={() => seekTo(seg.start)}
            >
              <span className="text-xs text-slate-400 w-20 shrink-0 pt-0.5 tabular-nums">
                {fmtTime(seg.start)}
              </span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <SpeakerEditPill
                    speakerId={seg.speaker}
                    displayName={spName(seg.speaker)}
                    colorClass={SPEAKER_COLORS[seg.speaker] ?? 'bg-slate-100 text-slate-600'}
                    gender={profile?.gender ?? null}
                    onRename={(name) => setSpeakerDisplayName?.(seg.speaker, name)}
                  />
                  {isOverlapping && (
                    <span
                      className="text-[10px] bg-orange-50 text-orange-500 px-1.5 py-0.5 rounded-full"
                      title={`Nakładanie głosów${seg.overlap_with ? ` z ${seg.overlap_with}` : ''}${seg.overlap_sec ? ` (${seg.overlap_sec.toFixed(1)}s)` : ''}`}
                    >
                      ⟨⟩ nakładanie
                    </span>
                  )}
                  {isInterjection && (
                    <span className="text-[10px] bg-violet-50 text-violet-500 px-1.5 py-0.5 rounded-full">wtrącenie</span>
                  )}
                  {isLowConf && (
                    <span className="text-[10px] bg-amber-50 text-amber-500 px-1.5 py-0.5 rounded-full">⚠ niska konf.</span>
                  )}
                  {hasIPA && (
                    <span className="text-[10px] bg-blue-50 text-blue-500 px-1.5 py-0.5 rounded-full font-mono">IPA</span>
                  )}
                  {hasAnyAlts && (
                    <span className="text-[10px] text-amber-600" title="Segment zawiera słowa z niską pewnością">✱ niepewne</span>
                  )}
                  {translationMissing && showTranslation && (
                    <span className="text-[10px] text-slate-400 italic">brak tłum.</span>
                  )}
                </div>

                {/* Polish translation */}
                {hasTranslation && showTranslation && (
                  <p className="text-slate-900 text-sm leading-relaxed mb-1">{seg.text_pl}</p>
                )}

                {/* Original text */}
                <p className={[
                  'text-sm leading-relaxed',
                  hasTranslation && showTranslation ? 'text-slate-400 text-xs italic' : 'text-slate-800',
                ].join(' ')}>
                  {hasTranslation && showTranslation && (
                    <span className="not-italic mr-1 opacity-60">({seg.language?.toUpperCase()})</span>
                  )}
                  {hasWords
                    ? seg.words!.map((w, wi) => (
                        w.text.trim().startsWith('[') || w.text.trim().startsWith('<')
                          ? <SoundBadge key={wi} text={w.text} />
                          : <WordChip key={wi} word={w} onAccept={(alt) => updateSegmentWord(seg.id, wi, alt)} />
                      ))
                    : <SoundBadge text={seg.text} />
                  }
                </p>

                {/* IPA annotations for low-confidence words */}
                {hasIPA && hasWords && (
                  <p className="mt-1 text-[10px] text-blue-400 font-mono">
                    {seg.words!.filter(w => (w as any).ipa).map((w, wi) => (
                      <span key={wi} className="mr-2">{w.text.trim()}→[{(w as any).ipa}]</span>
                    ))}
                  </p>
                )}
              </div>

              {/* Re-transcribe button (visible on hover) */}
              <button
                className="opacity-0 group-hover:opacity-100 transition-opacity shrink-0 p-1.5 rounded-lg text-slate-400 hover:text-brand-600 hover:bg-brand-50"
                title="Re-transkrybuj ten segment z lepszymi parametrami"
                onClick={async (e) => {
                  e.stopPropagation()
                  const agentSid = session.agentSessionId
                  if (!agentSid) { alert('Sesja agenta nieaktywna'); return }
                  try {
                    await fetch(`${settings.transcribeUrl}/hint/${agentSid}`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        hint: `Re-transcribe segment id=${seg.id} (${fmtTime(seg.start)}–${fmtTime(seg.end)}) with multi_pass_transcribe_segment. Use padding_sec=2.0 and high beam_size=8.`,
                      }),
                    })
                  } catch { alert('Nie można wysłać wskazówki do agenta') }
                }}
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
