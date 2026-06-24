import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, Mic, AlertCircle, Loader2, FileJson, Brain, Wrench, CheckCircle2, XCircle, BookMarked, ShieldAlert, Eye, ChevronDown, RefreshCw, Send } from 'lucide-react'
import { useSessionStore } from '../../store/sessionStore'
import { useSettingsStore } from '../../store/settingsStore'
import { useAgentPipeline } from '../../hooks/useAgentPipeline'
import type { Segment, AgentEvent } from '@pandaro/shared-types'

const ACCEPTED = ['.mp3', '.mp4', '.m4a', '.wav', 'audio/*', 'video/mp4']

const TOOL_LABELS: Record<string, string> = {
  probe_audio_fragment: 'Próbkowanie audio',
  detect_speaker_count: 'Wykrywanie mówców',
  set_transcription_params: 'Konfiguracja parametrów',
  analyze_audio_quality: 'Analiza jakości audio',
  detect_language_switches: 'Wykrywanie przełączeń języka',
  get_audio_info: 'Analiza audio',
  transcribe_audio: 'Transkrypcja (GPU)',
  verify_transcript_quality: 'Weryfikacja jakości transkryptu',
  split_long_segments: 'Podział długich segmentów',
  merge_short_segments: 'Scalanie krótkich segmentów',
  diarize_audio: 'Diaryzacja mówców',
  normalize_speaker_labels: 'Normalizacja etykiet mówców',
  profile_speakers: 'Analiza cech głosu',
  translate_to_polish: 'Tłumaczenie na polski (sub-agent)',
  validate_translation_quality: 'Ocena jakości tłumaczenia',
  retranslate_segments: 'Retranslacja słabych segmentów',
  identify_speakers: 'Rozpoznawanie imion',
  extract_entities: 'Ekstrakcja encji (LLM)',
  extract_keywords_statistical: 'Słowa kluczowe (TF-IDF)',
  detect_topics: 'Wykrywanie tematów',
  compute_text_statistics: 'Statystyki tekstu',
  search_in_transcript: 'Wyszukiwanie w transkrypcie',
  emit_partial_result: 'Podgląd wyników',
  build_rag_index: 'Budowanie indeksu RAG',
  summarize_transcript: 'Generowanie podsumowania',
  run_analysis: 'Analiza kodu',
  save_checkpoint: 'Zapis punktu kontrolnego',
  load_checkpoint: 'Wczytanie punktu kontrolnego',
  save_memory: 'Zapis do pamięci agenta',
  finish: 'Zakończenie',
  refine_speaker_assignments: 'Ulepszenie przypisań mówców',
  extract_quotes_and_facts: 'Ekstrakcja cytatów i faktów',
  verify_names_and_locations: 'Weryfikacja nazw i lokalizacji',
  multi_pass_transcribe_segment: 'Multi-pass transkrypcja segmentu',
}

const SPEAKER_COLORS: Record<string, string> = {
  GŁOS_01: 'bg-blue-100 text-blue-800', GŁOS_02: 'bg-violet-100 text-violet-800',
  GŁOS_03: 'bg-amber-100 text-amber-800', GŁOS_04: 'bg-green-100 text-green-800',
  GŁOS_05: 'bg-rose-100 text-rose-800',
}

function AgentEventRow({ event }: { event: AgentEvent }) {
  if (event.type === 'agent_thinking') {
    return (
      <li className="flex items-center gap-2 text-xs text-slate-400 py-0.5">
        <Brain className="w-3.5 h-3.5 shrink-0 text-brand-300" />
        <span>Agent analizuje sytuację…</span>
      </li>
    )
  }
  if (event.type === 'tool_call') {
    const label = TOOL_LABELS[event.tool ?? ''] ?? event.tool
    return (
      <li className="flex items-center gap-2 text-xs text-slate-500 py-0.5">
        <Wrench className="w-3.5 h-3.5 shrink-0 text-brand-400" />
        <span className="font-medium text-slate-700">{label}</span>
        {(event.attempt ?? 1) > 1 && (
          <span className="ml-auto text-amber-500">próba {event.attempt}</span>
        )}
      </li>
    )
  }
  if (event.type === 'tool_result') {
    const label = TOOL_LABELS[event.tool ?? ''] ?? event.tool
    const skipped = (event.result as any)?.skipped
    return (
      <li className="flex items-center gap-2 text-xs py-0.5">
        <CheckCircle2 className={`w-3.5 h-3.5 shrink-0 ${skipped ? 'text-slate-300' : 'text-green-500'}`} />
        <span className={skipped ? 'text-slate-400 line-through' : 'text-slate-600'}>{label}</span>
        {skipped && <span className="ml-1 text-slate-400 text-xs">(pominięto)</span>}
      </li>
    )
  }
  if (event.type === 'tool_error') {
    const label = TOOL_LABELS[event.tool ?? ''] ?? event.tool
    return (
      <li className="flex items-center gap-2 text-xs py-0.5">
        <XCircle className="w-3.5 h-3.5 shrink-0 text-amber-500" />
        <span className="text-amber-700">{label}</span>
        <span className="ml-auto text-amber-500 text-xs truncate max-w-[140px]">{event.error}</span>
      </li>
    )
  }
  if (event.type === 'quality_report') {
    const warnings = event.warnings ?? []
    if (warnings.length === 0) return null
    return (
      <li className="flex items-start gap-2 text-xs py-0.5">
        <ShieldAlert className="w-3.5 h-3.5 shrink-0 text-amber-500 mt-0.5" />
        <span className="text-amber-700">{warnings[0]}</span>
      </li>
    )
  }
  if ((event as any).type === 'translation_quality_check') {
    const avg = (event as any).batch_avg ?? 0
    const color = avg >= 4 ? 'text-green-600' : avg >= 3 ? 'text-amber-600' : 'text-red-600'
    return (
      <li className="flex items-center gap-2 text-xs py-0.5">
        <CheckCircle2 className={`w-3.5 h-3.5 shrink-0 ${color}`} />
        <span className={color}>Jakość tłumaczenia: {avg.toFixed(1)}/5</span>
        {((event as any).issues ?? []).length > 0 && (
          <span className="ml-1 text-slate-400 truncate max-w-[120px]">{(event as any).issues[0]}</span>
        )}
      </li>
    )
  }
  if (event.type === 'partial_segments') {
    return (
      <li className="flex items-center gap-2 text-xs py-0.5">
        <Eye className="w-3.5 h-3.5 shrink-0 text-brand-500" />
        <span className="text-brand-700 font-medium">Podgląd transkryptu dostępny</span>
      </li>
    )
  }
  if (event.type === 'agent_memory') {
    return (
      <li className="flex items-start gap-2 text-xs py-0.5">
        <BookMarked className="w-3.5 h-3.5 shrink-0 text-purple-400 mt-0.5" />
        <span className="text-purple-700 italic">{event.memory?.observation}</span>
      </li>
    )
  }
  return null
}

export default function UploadPage() {
  const [dragging, setDragging] = useState(false)
  const [jsonError, setJsonError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const jsonInputRef = useRef<HTMLInputElement>(null)
  const { process, cancel } = useAgentPipeline()
  const { session, clearSession, loadTranscript } = useSessionStore()
  const { settings, update: updateSettings } = useSettingsStore()
  const navigate = useNavigate()

  // ── Hint input state ──────────────────────────────────────────────────────
  const [hintText, setHintText] = useState('')
  const [hintSent, setHintSent] = useState(false)

  const sendHint = useCallback(async () => {
    const sid = session?.agentSessionId
    if (!sid || !hintText.trim()) return
    try {
      await fetch(`${settings.transcribeUrl}/session/${sid}/hint`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hint: hintText.trim() }),
      })
      setHintSent(true)
      setHintText('')
      setTimeout(() => setHintSent(false), 2000)
    } catch { /* non-fatal */ }
  }, [session?.agentSessionId, hintText, settings.transcribeUrl])
  const [models, setModels] = useState<string[]>([])
  const [modelsLoading, setModelsLoading] = useState(false)

  const fetchModels = useCallback(async () => {
    setModelsLoading(true)
    try {
      const res = await fetch(`${settings.transcribeUrl}/models`)
      if (res.ok) {
        const data = await res.json()
        setModels(data.models ?? [])
      }
    } catch { /* Ollama not reachable — show empty list */ }
    finally { setModelsLoading(false) }
  }, [settings.transcribeUrl])

  useEffect(() => { fetchModels() }, [fetchModels])

  const isProcessing =
    session !== null &&
    session.processing.step !== 'idle' &&
    session.processing.step !== 'done' &&
    session.processing.step !== 'error'

  const handleFile = useCallback(
    (file: File) => {
      clearSession()
      process(file, 'whisper')
    },
    [process, clearSession],
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  const handleJsonFile = useCallback(
    (file: File) => {
      setJsonError(null)
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const raw = JSON.parse(e.target?.result as string)
          const segsRaw: any[] = Array.isArray(raw)
            ? raw
            : Array.isArray(raw.segments)
            ? raw.segments
            : null
          if (!segsRaw) throw new Error('Nie znaleziono tablicy segmentów w pliku JSON.')
          const segments: Segment[] = segsRaw.map((s: any, i: number) => ({
            id: i,
            start: Number(s.start ?? 0),
            end: Number(s.end ?? 0),
            text: String(s.text ?? ''),
            text_pl: s.text_pl ? String(s.text_pl) : undefined,
            speaker: String(s.speaker ?? 'GŁOS_01'),
            language: s.language ? String(s.language) : undefined,
          }))
          const detectedLanguage = raw.detected_language ?? raw.language ?? 'auto'
          loadTranscript(file.name.replace(/\.json$/, ''), segments, detectedLanguage)
          navigate('/transcript')
        } catch (err: any) {
          setJsonError(err?.message ?? 'Błąd parsowania pliku JSON.')
        }
      }
      reader.readAsText(file)
    },
    [loadTranscript, navigate],
  )

  // Show only the most recent meaningful agent events (last 12)
  const agentEvents: AgentEvent[] = session?.agentEvents ?? []
  const visibleEvents = agentEvents.filter(
    (e: AgentEvent) => e.type !== 'result' && e.type !== 'agent_start',
  ).slice(-12)

  // Partial transcript preview (first 5 segments shown during processing)
  const hasPartial = isProcessing && (session?.segments?.length ?? 0) > 0
  const previewSegs = session?.segments?.slice(0, 5) ?? []

  // Quality warnings
  const qualityWarnings = session?.qualityStats?.warnings ?? []

  return (
    <div className="flex flex-col items-center justify-center min-h-full px-6 py-12">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-brand-50 rounded-2xl mb-4">
            <Mic className="w-8 h-8 text-brand-600" />
          </div>
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Pandaro</h1>
          <p className="text-slate-500 text-lg">
            Transkrypcja, diaryzacja i analiza AI nagrań audio
          </p>
        </div>

        {/* Model selector */}
        {!isProcessing && (
          <div className="mb-6 bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <div className="flex items-center gap-2 mb-3">
              <Brain className="w-4 h-4 text-brand-600 shrink-0" />
              <span className="text-sm font-semibold text-slate-800">Model LLM (Ollama)</span>
              <button
                onClick={fetchModels}
                disabled={modelsLoading}
                className="ml-auto text-slate-400 hover:text-brand-600 transition-colors disabled:opacity-40"
                title="Odśwież listę modeli"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${modelsLoading ? 'animate-spin' : ''}`} />
              </button>
            </div>
            {models.length === 0 ? (
              <div className="flex items-center gap-2 text-xs text-slate-500">
                {modelsLoading ? (
                  <><Loader2 className="w-3.5 h-3.5 animate-spin" /><span>Ładowanie modeli…</span></>
                ) : (
                  <span className="text-amber-600">Brak połączenia z Ollama — model: <strong>{settings.ollamaModel}</strong></span>
                )}
              </div>
            ) : (
              <div className="relative">
                <select
                  value={settings.ollamaModel}
                  onChange={(e) => updateSettings({ ollamaModel: e.target.value })}
                  className="w-full appearance-none bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 pr-8 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent"
                >
                  {models.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                  {/* If current model isn't in list (e.g. env default), add it */}
                  {settings.ollamaModel && !models.includes(settings.ollamaModel) && (
                    <option value={settings.ollamaModel}>{settings.ollamaModel} (ustawienie domyślne)</option>
                  )}
                </select>
                <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400" />
              </div>
            )}
          </div>
        )}

        {/* Drop Zone */}
        {!isProcessing && (
          <div
            className={[
              'border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-colors',
              dragging
                ? 'border-brand-400 bg-brand-50'
                : 'border-slate-200 hover:border-brand-300 hover:bg-slate-50',
            ].join(' ')}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
          >
            <Upload className="w-10 h-10 text-slate-300 mx-auto mb-4" />
            <p className="text-slate-700 font-medium mb-1">
              Przeciągnij plik tutaj lub kliknij, aby wybrać
            </p>
            <p className="text-sm text-slate-400">MP3, MP4, M4A, WAV</p>
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED.join(',')}
              className="hidden"
              onChange={onInputChange}
            />
          </div>
        )}

        {/* Load from JSON */}
        {!isProcessing && (
          <div className="mt-4">
            <button
              onClick={() => jsonInputRef.current?.click()}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl border border-slate-200 text-sm text-slate-600 hover:bg-slate-50 hover:border-brand-300 transition-colors"
            >
              <FileJson className="w-4 h-4 text-slate-400" />
              Załaduj transkrypcję z pliku JSON
            </button>
            <input
              ref={jsonInputRef}
              type="file"
              accept=".json,application/json"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) handleJsonFile(f)
                e.target.value = ''
              }}
            />
            {jsonError && (
              <p className="mt-2 text-xs text-red-600 text-center">{jsonError}</p>
            )}
          </div>
        )}

        {/* Agent Activity Panel */}
        {session && isProcessing && (
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
            {/* Header */}
            <div className="flex items-center gap-3 mb-4">
              <Loader2 className="w-5 h-5 text-brand-600 animate-spin shrink-0" />
              <div className="min-w-0 flex-1">
                <p className="font-medium text-slate-800 truncate">{session.fileName}</p>
                <p className="text-sm text-slate-500">{session.processing.message}</p>
              </div>
              <span className="shrink-0 text-xs text-brand-600 bg-brand-50 border border-brand-200 rounded-full px-2 py-0.5 font-medium">
                {settings.ollamaModel}
              </span>
            </div>

            {/* Progress bar */}
            <div className="w-full bg-slate-100 rounded-full h-1.5 mb-5">
              <div
                className="bg-brand-600 h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${session.processing.progress}%` }}
              />
            </div>

            {/* Agent activity log */}
            <div className="bg-slate-50 rounded-xl border border-slate-100 p-3 min-h-[140px]">
              <p className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wide">
                Aktywność agenta
              </p>
              {visibleEvents.length === 0 ? (
                <p className="text-xs text-slate-300 italic">Inicjalizacja…</p>
              ) : (
                <ul className="space-y-0.5">
                  {visibleEvents.map((ev: AgentEvent, i: number) => (
                    <AgentEventRow key={i} event={ev} />
                  ))}
                </ul>
              )}
            </div>

            <button
              onClick={cancel}
              className="mt-4 w-full text-sm text-slate-500 hover:text-red-600 underline hover:no-underline"
            >
              Anuluj
            </button>

            {/* Human hint input */}
            <div className="mt-3 flex gap-2">
              <input
                type="text"
                value={hintText}
                onChange={(e) => setHintText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendHint()}
                placeholder="Wskazówka dla agenta (np. 'To wywiad z 2 osobami po rosyjsku')"
                className="flex-1 text-xs border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-500 bg-white"
              />
              <button
                onClick={sendHint}
                disabled={!hintText.trim() || !session?.agentSessionId}
                className="shrink-0 bg-brand-600 disabled:bg-slate-200 text-white disabled:text-slate-400 rounded-lg px-3 py-2 text-xs font-medium hover:bg-brand-700 transition-colors"
              >
                {hintSent ? '✓' : <Send className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>
        )}

        {/* Quality warnings */}
        {isProcessing && qualityWarnings.length > 0 && (
          <div className="mt-3 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3">
            <p className="text-xs font-semibold text-amber-800 mb-1 flex items-center gap-1.5">
              <ShieldAlert className="w-3.5 h-3.5" /> Jakość transkrypcji
            </p>
            <ul className="space-y-0.5">
              {qualityWarnings.slice(0, 3).map((w, i) => (
                <li key={i} className="text-xs text-amber-700">{w}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Partial transcript preview */}
        {hasPartial && previewSegs.length > 0 && (
          <div className="mt-3 bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <p className="text-xs font-semibold text-slate-500 mb-2 flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5 text-brand-500" />
              Podgląd transkryptu
              {session?.segmentsPartial && (
                <span className="ml-auto text-brand-400 animate-pulse">wstępny</span>
              )}
            </p>
            <div className="space-y-1.5">
              {previewSegs.map((seg: Segment) => (
                <div key={seg.id} className="flex gap-2 text-xs">
                  <span className={`shrink-0 px-1.5 py-0.5 rounded text-xs font-medium ${SPEAKER_COLORS[seg.speaker] ?? 'bg-slate-100 text-slate-600'}`}>
                    {seg.speaker}
                  </span>
                  <span className="text-slate-700 leading-relaxed">{seg.text_pl ?? seg.text}</span>
                </div>
              ))}
              {(session?.segments?.length ?? 0) > 5 && (
                <p className="text-xs text-slate-400 pl-1">…+{(session?.segments?.length ?? 0) - 5} segmentów</p>
              )}
            </div>
          </div>
        )}

        {/* Error */}
        {session?.processing.step === 'error' && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-red-800">Wystąpił błąd</p>
              <p className="text-sm text-red-600">{session.processing.error}</p>
              <button
                onClick={() => clearSession()}
                className="mt-2 text-sm text-red-700 underline hover:no-underline"
              >
                Spróbuj ponownie
              </button>
            </div>
          </div>
        )}

        {!isProcessing && (
          <p className="text-center text-xs text-slate-400 mt-6">
            Przetwarzanie odbywa się na serwerze GPU. Agent AI orkiestruje wszystkie etapy.
          </p>
        )}
      </div>
    </div>
  )
}
