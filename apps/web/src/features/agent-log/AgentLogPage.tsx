import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Brain, Wrench, CheckCircle2, XCircle, BookMarked, ChevronDown, ChevronRight,
  ShieldAlert, Eye, Cpu, MessageSquare, Database, ArrowLeft,
} from 'lucide-react'
import { useSessionStore } from '../../store/sessionStore'
import type { AgentEvent } from '@pandaro/shared-types'

const TOOL_LABELS: Record<string, string> = {
  probe_audio_fragment: 'Próbkowanie audio',
  detect_speaker_count: 'Wykrywanie mówców',
  set_transcription_params: 'Konfiguracja parametrów',
  analyze_audio_quality: 'Analiza jakości audio',
  detect_language_switches: 'Wykrywanie przełączeń języka',
  compare_transcription_params: 'Porównanie parametrów',
  get_audio_info: 'Analiza audio',
  transcribe_audio: 'Transkrypcja (GPU)',
  verify_transcript_quality: 'Weryfikacja jakości',
  split_long_segments: 'Podział segmentów',
  merge_short_segments: 'Scalanie segmentów',
  diarize_audio: 'Diaryzacja mówców',
  normalize_speaker_labels: 'Normalizacja etykiet',
  profile_speakers: 'Analiza cech głosu',
  translate_to_polish: 'Tłumaczenie na polski',
  validate_translation_quality: 'Ocena jakości tłumaczenia',
  retranslate_segments: 'Retranslacja',
  retranscribe_time_range: 'Re-transkrypcja fragmentu',
  identify_speakers: 'Rozpoznawanie imion',
  extract_entities: 'Ekstrakcja encji',
  extract_keywords_statistical: 'Słowa kluczowe',
  detect_topics: 'Wykrywanie tematów',
  compute_text_statistics: 'Statystyki tekstu',
  search_in_transcript: 'Wyszukiwanie',
  emit_partial_result: 'Podgląd wyników',
  build_rag_index: 'Indeks RAG',
  summarize_transcript: 'Podsumowanie',
  run_analysis: 'Analiza kodu',
  write_artifact: 'Zapis artefaktu',
  read_artifact: 'Odczyt artefaktu',
  list_artifacts: 'Lista artefaktów',
  save_checkpoint: 'Checkpoint',
  load_checkpoint: 'Wczytanie checkpointu',
  save_memory: 'Pamięć agenta',
  finish: 'Zakończenie',
}

const TOOL_ICONS: Record<string, typeof Brain> = {
  transcribe_audio: Cpu,
  diarize_audio: Brain,
  translate_to_polish: MessageSquare,
  save_memory: BookMarked,
  write_artifact: Database,
  read_artifact: Database,
  emit_partial_result: Eye,
  verify_transcript_quality: ShieldAlert,
  validate_translation_quality: ShieldAlert,
}

function EventRow({ event, index }: { event: AgentEvent; index: number }) {
  const [expanded, setExpanded] = useState(false)

  const type = event.type

  if (type === 'agent_thinking') {
    return (
      <div className="flex items-center gap-2 py-1 text-xs text-slate-400 pl-2">
        <Brain className="w-3.5 h-3.5 shrink-0 text-brand-300" />
        <span>Krok {(event as any).step ?? index} — agent decyduje…</span>
      </div>
    )
  }

  if (type === 'hint_injected') {
    return (
      <div className="flex items-center gap-2 py-1 text-xs pl-2">
        <MessageSquare className="w-3.5 h-3.5 shrink-0 text-purple-400" />
        <span className="text-purple-700 font-medium">Wskazówka:</span>
        <span className="text-purple-600">{(event as any).hint}</span>
      </div>
    )
  }

  if (type === 'agent_memory') {
    return (
      <div className="py-1 pl-2">
        <div className="flex items-start gap-2 text-xs">
          <BookMarked className="w-3.5 h-3.5 shrink-0 text-purple-400 mt-0.5" />
          <div>
            <span className="font-medium text-purple-700">Pamięć zapisana</span>
            <p className="text-purple-600 italic mt-0.5">{event.memory?.observation}</p>
          </div>
        </div>
      </div>
    )
  }

  if (type === 'tool_call') {
    const label = TOOL_LABELS[event.tool ?? ''] ?? event.tool
    const Icon = TOOL_ICONS[event.tool ?? ''] ?? Wrench
    const args = event.args
    const hasArgs = args && Object.keys(args).length > 0
    return (
      <div className="py-0.5 pl-2">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-xs w-full text-left hover:bg-slate-50 rounded px-1 py-0.5"
        >
          {hasArgs
            ? (expanded ? <ChevronDown className="w-3 h-3 text-slate-400" /> : <ChevronRight className="w-3 h-3 text-slate-400" />)
            : <span className="w-3" />
          }
          <Icon className="w-3.5 h-3.5 shrink-0 text-brand-500" />
          <span className="font-medium text-slate-700">{label}</span>
          {(event.attempt ?? 1) > 1 && (
            <span className="ml-auto text-amber-500 text-xs">pokusa {event.attempt}</span>
          )}
        </button>
        {expanded && hasArgs && (
          <pre className="ml-6 mt-1 text-xs bg-slate-50 rounded p-2 text-slate-600 overflow-x-auto">
            {JSON.stringify(args, null, 2)}
          </pre>
        )}
      </div>
    )
  }

  if (type === 'tool_result') {
    const label = TOOL_LABELS[event.tool ?? ''] ?? event.tool
    const skipped = (event.result as any)?.skipped
    const result = event.result
    const hasResult = result && Object.keys(result).length > 0
    return (
      <div className="py-0.5 pl-2">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-xs w-full text-left hover:bg-slate-50 rounded px-1 py-0.5"
        >
          {hasResult
            ? (expanded ? <ChevronDown className="w-3 h-3 text-slate-400" /> : <ChevronRight className="w-3 h-3 text-slate-400" />)
            : <span className="w-3" />
          }
          <CheckCircle2 className={`w-3.5 h-3.5 shrink-0 ${skipped ? 'text-slate-300' : 'text-green-500'}`} />
          <span className={`${skipped ? 'text-slate-400 line-through' : 'text-slate-700'}`}>{label}</span>
          {skipped && <span className="text-slate-400 ml-1">(pominięto)</span>}
          {!skipped && result && (
            <span className="ml-auto text-slate-400 truncate max-w-[200px]">
              {Object.entries(result as object).slice(0, 2).map(([k, v]) => `${k}: ${v}`).join(' · ')}
            </span>
          )}
        </button>
        {expanded && hasResult && (
          <pre className="ml-6 mt-1 text-xs bg-green-50 rounded p-2 text-green-800 overflow-x-auto">
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </div>
    )
  }

  if (type === 'tool_error') {
    const label = TOOL_LABELS[event.tool ?? ''] ?? event.tool
    return (
      <div className="py-0.5 pl-2">
        <div className="flex items-center gap-2 text-xs px-1 py-0.5">
          <XCircle className="w-3.5 h-3.5 shrink-0 text-red-400" />
          <span className="text-red-700 font-medium">{label}</span>
          <span className="text-red-500 truncate max-w-[240px]">{event.error}</span>
          {(event.attempt ?? 1) > 1 && <span className="text-amber-500">próba {event.attempt}</span>}
        </div>
      </div>
    )
  }

  if (type === 'quality_report') {
    const warnings = (event as any).warnings ?? []
    const avg = (event as any).stats?.avg_confidence
    return (
      <div className="py-0.5 pl-2">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-xs w-full text-left hover:bg-slate-50 rounded px-1 py-0.5"
        >
          {expanded ? <ChevronDown className="w-3 h-3 text-slate-400" /> : <ChevronRight className="w-3 h-3 text-slate-400" />}
          <ShieldAlert className="w-3.5 h-3.5 shrink-0 text-amber-500" />
          <span className="text-amber-700 font-medium">Raport jakości</span>
          {avg !== undefined && (
            <span className={`ml-auto text-xs ${avg >= 0.75 ? 'text-green-600' : avg >= 0.60 ? 'text-amber-600' : 'text-red-600'}`}>
              konfidencja: {(avg * 100).toFixed(0)}%
            </span>
          )}
        </button>
        {expanded && warnings.length > 0 && (
          <ul className="ml-6 mt-1 space-y-0.5">
            {warnings.map((w: string, i: number) => (
              <li key={i} className="text-xs text-amber-700">⚠ {w}</li>
            ))}
          </ul>
        )}
      </div>
    )
  }

  return null
}

export default function AgentLogPage() {
  const { session } = useSessionStore()
  const navigate = useNavigate()

  if (!session) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-400">
        <Brain className="w-12 h-12 mb-4" />
        <p>Brak aktywnej sesji agenta.</p>
        <button onClick={() => navigate('/')} className="mt-4 text-brand-600 underline text-sm">
          Wróć do głównej
        </button>
      </div>
    )
  }

  const events = session.agentEvents.filter(
    (e) => e.type !== 'agent_start' && e.type !== 'result' &&
           e.type !== 'segment_chunk' && e.type !== 'translation_chunk' &&
           e.type !== 'partial_segments' && e.type !== 'progress'
  )

  const isProcessing =
    session.processing.step !== 'idle' &&
    session.processing.step !== 'done' &&
    session.processing.step !== 'error'

  const toolCalls = events.filter((e) => e.type === 'tool_call').length
  const toolErrors = events.filter((e) => e.type === 'tool_error').length
  const memories = events.filter((e) => e.type === 'agent_memory').length

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-6 py-4 flex items-center gap-4">
        <button onClick={() => navigate(-1)} className="text-slate-400 hover:text-slate-700">
          <ArrowLeft className="w-4 h-4" />
        </button>
        <Brain className="w-5 h-5 text-brand-600" />
        <div className="flex-1">
          <h1 className="font-semibold text-slate-900">Log agenta</h1>
          <p className="text-xs text-slate-500">{session.fileName}</p>
        </div>
        {isProcessing && (
          <span className="flex items-center gap-1.5 text-xs text-brand-600 bg-brand-50 px-2 py-1 rounded-full">
            <span className="w-1.5 h-1.5 bg-brand-500 rounded-full animate-pulse" />
            Przetwarzanie…
          </span>
        )}
      </div>

      {/* Stats bar */}
      <div className="bg-slate-50 border-b border-slate-100 px-6 py-2 flex gap-6 text-xs text-slate-500">
        <span><strong className="text-slate-700">{events.length}</strong> zdarzeń</span>
        <span><strong className="text-slate-700">{toolCalls}</strong> wywołań narzędzi</span>
        {toolErrors > 0 && <span><strong className="text-red-600">{toolErrors}</strong> błędów</span>}
        {memories > 0 && <span><strong className="text-purple-600">{memories}</strong> pamięci</span>}
        {session.agentSessionId && (
          <span className="ml-auto font-mono text-slate-300">id: {session.agentSessionId}</span>
        )}
      </div>

      {/* Event log */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {events.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-slate-300">
            <Brain className="w-10 h-10 mb-3" />
            <p className="text-sm">Brak zdarzeń agenta</p>
          </div>
        ) : (
          <div className="space-y-0.5 divide-y divide-slate-50">
            {events.map((event, i) => (
              <EventRow key={i} event={event} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
