import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Brain, Wrench, CheckCircle2, XCircle, BookMarked, ChevronDown, ChevronRight,
  ShieldAlert, Eye, Cpu, MessageSquare, Database, ArrowLeft, AlertTriangle,
  Copy, Check, Zap, Layers, Tag, Mic2,
} from 'lucide-react'
import { useSessionStore } from '../../store/sessionStore'
import type { AgentEvent } from '@pandaro/shared-types'

const TOOL_META: Record<string, { label: string; icon: typeof Brain; color: string; category: string }> = {
  get_audio_info:              { label: 'Analiza audio',            icon: Zap,          color: 'text-slate-500',  category: 'Kalibracja' },
  analyze_audio_quality:       { label: 'Jakość sygnału',           icon: Zap,          color: 'text-slate-500',  category: 'Kalibracja' },
  detect_noise_regions:        { label: 'Regiony ciszy/szumu',      icon: Zap,          color: 'text-slate-500',  category: 'Kalibracja' },
  probe_audio_fragment:        { label: 'Próbkowanie audio',        icon: Zap,          color: 'text-blue-500',   category: 'Kalibracja' },
  set_transcription_params:    { label: 'Konfiguracja parametrów',  icon: Wrench,       color: 'text-blue-500',   category: 'Kalibracja' },
  compare_transcription_params:{ label: 'Porównanie parametrów',    icon: Wrench,       color: 'text-blue-500',   category: 'Kalibracja' },
  detect_speaker_count:        { label: 'Wykrywanie mówców',        icon: Mic2,         color: 'text-violet-500', category: 'Diaryzacja' },
  detect_language_switches:    { label: 'Przełączenia języka',      icon: MessageSquare,color: 'text-blue-500',   category: 'Kalibracja' },
  transcribe_audio:            { label: 'Transkrypcja (GPU)',       icon: Cpu,          color: 'text-brand-600',  category: 'Transkrypcja' },
  verify_transcript_quality:   { label: 'Weryfikacja jakości',      icon: ShieldAlert,  color: 'text-amber-500',  category: 'Transkrypcja' },
  tag_segments:                { label: 'Tagowanie segmentów',      icon: Tag,          color: 'text-slate-500',  category: 'Transkrypcja' },
  split_long_segments:         { label: 'Podział segmentów',        icon: Layers,       color: 'text-slate-500',  category: 'Transkrypcja' },
  merge_short_segments:        { label: 'Scalanie segmentów',       icon: Layers,       color: 'text-slate-500',  category: 'Transkrypcja' },
  multi_pass_transcribe_segment:{ label: 'Re-transkrypcja segmentu',icon: Cpu,          color: 'text-amber-500',  category: 'Transkrypcja' },
  retranscribe_time_range:     { label: 'Re-transkrypcja fragmentu',icon: Cpu,          color: 'text-amber-500',  category: 'Transkrypcja' },
  diarize_audio:               { label: 'Diaryzacja mówców',        icon: Brain,        color: 'text-violet-600', category: 'Diaryzacja' },
  refine_speaker_assignments:  { label: 'Doprecyzowanie mówców',    icon: Mic2,         color: 'text-violet-500', category: 'Diaryzacja' },
  merge_duplicate_speakers:    { label: 'Scalanie duplikatów',      icon: Mic2,         color: 'text-violet-500', category: 'Diaryzacja' },
  normalize_speaker_labels:    { label: 'Normalizacja etykiet',     icon: Mic2,         color: 'text-violet-400', category: 'Diaryzacja' },
  profile_speakers:            { label: 'Profil głosu',             icon: Mic2,         color: 'text-violet-500', category: 'Diaryzacja' },
  identify_speakers:           { label: 'Identyfikacja mówców',     icon: Mic2,         color: 'text-violet-600', category: 'Diaryzacja' },
  translate_to_polish:         { label: 'Tłumaczenie → PL',         icon: MessageSquare,color: 'text-emerald-600',category: 'Tłumaczenie' },
  validate_translation_quality:{ label: 'Ocena tłumaczenia',        icon: ShieldAlert,  color: 'text-amber-500',  category: 'Tłumaczenie' },
  retranslate_segments:        { label: 'Ponowne tłumaczenie',      icon: MessageSquare,color: 'text-emerald-500',category: 'Tłumaczenie' },
  emit_partial_result:         { label: 'Podgląd wyników',          icon: Eye,          color: 'text-blue-500',   category: 'UI' },
  extract_entities:            { label: 'Ekstrakcja encji',         icon: Database,     color: 'text-indigo-500', category: 'Analiza' },
  verify_names_and_locations:  { label: 'Weryfikacja nazw',         icon: ShieldAlert,  color: 'text-indigo-400', category: 'Analiza' },
  extract_quotes_and_facts:    { label: 'Cytaty i fakty',           icon: BookMarked,   color: 'text-indigo-500', category: 'Analiza' },
  extract_keywords_statistical:{ label: 'Słowa kluczowe',           icon: Database,     color: 'text-indigo-400', category: 'Analiza' },
  detect_topics:               { label: 'Wykrywanie tematów',       icon: Database,     color: 'text-indigo-500', category: 'Analiza' },
  compute_text_statistics:     { label: 'Statystyki tekstu',        icon: Database,     color: 'text-slate-500',  category: 'Analiza' },
  run_analysis:                { label: 'Analiza kodu',             icon: Cpu,          color: 'text-slate-500',  category: 'Analiza' },
  build_rag_index:             { label: 'Indeks RAG',               icon: Database,     color: 'text-blue-600',   category: 'Synteza' },
  summarize_transcript:        { label: 'Podsumowanie',             icon: BookMarked,   color: 'text-blue-600',   category: 'Synteza' },
  save_checkpoint:             { label: 'Checkpoint',               icon: Database,     color: 'text-slate-400',  category: 'Kontekst' },
  load_checkpoint:             { label: 'Wczyt. checkpointu',       icon: Database,     color: 'text-slate-400',  category: 'Kontekst' },
  write_artifact:              { label: 'Zapis artefaktu',          icon: Database,     color: 'text-slate-400',  category: 'Kontekst' },
  read_artifact:               { label: 'Odczyt artefaktu',         icon: Database,     color: 'text-slate-400',  category: 'Kontekst' },
  list_artifacts:              { label: 'Lista artefaktów',         icon: Database,     color: 'text-slate-400',  category: 'Kontekst' },
  search_in_transcript:        { label: 'Wyszukiwanie',             icon: Eye,          color: 'text-slate-500',  category: 'UI' },
  save_memory:                 { label: 'Pamięć agenta',            icon: BookMarked,   color: 'text-purple-500', category: 'Pamięć' },
  finish:                      { label: 'Zakończenie',              icon: CheckCircle2, color: 'text-green-600',  category: 'Kontrola' },
}

const CATEGORY_COLORS: Record<string, string> = {
  Kalibracja: 'bg-blue-50 text-blue-700', Transkrypcja: 'bg-brand-50 text-brand-700',
  Diaryzacja: 'bg-violet-50 text-violet-700', Tłumaczenie: 'bg-emerald-50 text-emerald-700',
  Analiza: 'bg-indigo-50 text-indigo-700', Synteza: 'bg-sky-50 text-sky-700',
  Kontekst: 'bg-slate-100 text-slate-500', Pamięć: 'bg-purple-50 text-purple-700',
  UI: 'bg-slate-50 text-slate-500', Kontrola: 'bg-green-50 text-green-700',
}

function getMeta(tool: string) {
  return TOOL_META[tool] ?? { label: tool, icon: Wrench, color: 'text-slate-500', category: 'Inne' }
}

function JsonBlock({ data, bg = 'bg-slate-50' }: { data: unknown; bg?: string }) {
  const [copied, setCopied] = useState(false)
  const text = JSON.stringify(data, null, 2)
  return (
    <div className={`relative mt-1.5 rounded-lg overflow-hidden border border-slate-100 ${bg}`}>
      <button onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1500) }}
        className="absolute top-1.5 right-1.5 p-1 rounded text-slate-400 hover:text-slate-700 hover:bg-white/60">
        {copied ? <Check className="w-3 h-3 text-green-500" /> : <Copy className="w-3 h-3" />}
      </button>
      <pre className="text-[11px] p-2.5 pr-8 text-slate-700 overflow-x-auto whitespace-pre-wrap max-h-64 leading-relaxed">{text}</pre>
    </div>
  )
}

function ResultSummary({ tool, result }: { tool: string; result: Record<string, unknown> }) {
  const items: string[] = []
  if (tool === 'transcribe_audio') {
    if (result.segment_count != null) items.push(`${result.segment_count} segm.`)
    if (result.detected_language) items.push(`${result.detected_language}`)
    if (result.engine_used) items.push(String(result.engine_used))
  } else if (tool === 'diarize_audio') {
    if (result.speaker_count != null) items.push(`${result.speaker_count} mówców`)
  } else if (tool === 'verify_transcript_quality') {
    if (result.avg_confidence != null) items.push(`${((result.avg_confidence as number)*100).toFixed(0)}% konf.`)
    if (result.low_confidence_count != null) items.push(`niskich: ${result.low_confidence_count}`)
  } else if (tool === 'merge_duplicate_speakers') {
    const mp = (result.merged_pairs as any[]) ?? []
    const sk = (result.skipped_merges as any[]) ?? []
    items.push(mp.length > 0 ? `scalono ${mp.length}` : 'brak duplikatów')
    if (sk.length) items.push(`zablok. ${sk.length}`)
  } else if (tool === 'probe_audio_fragment') {
    if (result.avg_confidence != null) items.push(`${((result.avg_confidence as number)*100).toFixed(0)}% konf.`)
    if (result.detected_language) items.push(String(result.detected_language))
    if (result.fragments_probed) items.push(`${result.fragments_probed} fragm.`)
  } else if (tool === 'tag_segments') {
    if (result.segments_tagged) items.push(`otagowano: ${result.segments_tagged}`)
    const csi = (result.check_speaker_ids as any[]) ?? []
    if (csi.length) items.push(`sprawdź: ${csi.length}`)
  } else if (result.skipped) {
    items.push('pominięto'); if (result.reason) items.push(String(result.reason))
  } else {
    Object.entries(result).slice(0, 3).forEach(([k, v]) => {
      if (typeof v !== 'object' && v != null) items.push(`${k}: ${v}`)
    })
  }
  if (!items.length) return null
  return <span className="ml-auto text-slate-400 text-[10px] truncate max-w-[200px]">{items.join(' · ')}</span>
}

function ToolCallRow({ event }: { event: AgentEvent }) {
  const [expanded, setExpanded] = useState(false)
  const meta = getMeta(event.tool ?? '')
  const Icon = meta.icon
  const args = event.args
  const hasArgs = args && Object.keys(args).length > 0
  const alias = (event as any).original_name
  const attempt = event.attempt ?? 1
  return (
    <div className="py-0.5 pl-1 border-l-2 border-brand-100 ml-1">
      <button onClick={() => hasArgs && setExpanded(!expanded)}
        className="flex items-center gap-2 text-xs w-full text-left hover:bg-slate-50 rounded px-1.5 py-1">
        {hasArgs ? (expanded ? <ChevronDown className="w-3 h-3 text-slate-400 shrink-0" /> : <ChevronRight className="w-3 h-3 text-slate-400 shrink-0" />)
          : <span className="w-3 shrink-0" />}
        <Icon className={`w-3.5 h-3.5 shrink-0 ${meta.color}`} />
        <span className="font-semibold text-slate-800">{meta.label}</span>
        <span className={`text-[10px] px-1.5 py-0.5 rounded-full shrink-0 ${CATEGORY_COLORS[meta.category] ?? 'bg-slate-100 text-slate-500'}`}>{meta.category}</span>
        <code className="text-[10px] text-slate-400 font-mono">{event.tool}</code>
        {alias && <span className="text-[10px] text-amber-500">(alias: {alias})</span>}
        {attempt > 1 && <span className="ml-auto text-[10px] text-amber-500 bg-amber-50 px-1.5 py-0.5 rounded-full shrink-0">próba {attempt}</span>}
      </button>
      {hasArgs && !expanded && (
        <div className="ml-8 text-[10px] text-slate-500 font-mono truncate">
          {Object.entries(args as object).slice(0, 4).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join('  ')}{Object.keys(args as object).length > 4 ? ' …' : ''}
        </div>
      )}
      {expanded && hasArgs && <div className="ml-8"><p className="text-[10px] text-slate-400 font-medium uppercase tracking-wide mb-0.5">Parametry wejściowe</p><JsonBlock data={args} bg="bg-blue-50" /></div>}
    </div>
  )
}

function ToolResultRow({ event }: { event: AgentEvent }) {
  const [expanded, setExpanded] = useState(false)
  const meta = getMeta(event.tool ?? '')
  const result = event.result as Record<string, unknown> | undefined
  const skipped = Boolean(result?.skipped)
  const hasResult = result && Object.keys(result).filter(k => k !== 'skipped').length > 0
  return (
    <div className="py-0.5 pl-1 border-l-2 border-green-100 ml-1">
      <button onClick={() => hasResult && setExpanded(!expanded)}
        className="flex items-center gap-2 text-xs w-full text-left hover:bg-slate-50 rounded px-1.5 py-1">
        {hasResult ? (expanded ? <ChevronDown className="w-3 h-3 text-slate-400 shrink-0" /> : <ChevronRight className="w-3 h-3 text-slate-400 shrink-0" />) : <span className="w-3 shrink-0" />}
        <CheckCircle2 className={`w-3.5 h-3.5 shrink-0 ${skipped ? 'text-slate-300' : 'text-green-500'}`} />
        <span className={`font-medium ${skipped ? 'text-slate-400 line-through' : 'text-slate-700'}`}>{meta.label}</span>
        {skipped && <span className="text-[10px] text-slate-400">(pominięto)</span>}
        {!skipped && result && <ResultSummary tool={event.tool ?? ''} result={result as Record<string, unknown>} />}
      </button>
      {expanded && hasResult && result && <div className="ml-8"><p className="text-[10px] text-slate-400 font-medium uppercase tracking-wide mb-0.5">Wynik</p><JsonBlock data={result} bg="bg-green-50" /></div>}
    </div>
  )
}

function ToolErrorRow({ event }: { event: AgentEvent }) {
  const [expanded, setExpanded] = useState(false)
  const meta = getMeta(event.tool ?? '')
  const isUnknown = event.error?.startsWith('Unknown tool') || event.error?.startsWith('Did you mean')
  return (
    <div className="py-0.5 pl-1 border-l-2 border-red-200 ml-1">
      <button onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-xs w-full text-left hover:bg-red-50 rounded px-1.5 py-1">
        {expanded ? <ChevronDown className="w-3 h-3 text-red-400 shrink-0" /> : <ChevronRight className="w-3 h-3 text-red-400 shrink-0" />}
        <XCircle className="w-3.5 h-3.5 shrink-0 text-red-400" />
        <span className="font-medium text-red-700">{meta.label}</span>
        {isUnknown && <span className="text-[10px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded-full">nieznane narzędzie</span>}
        <span className="ml-auto text-red-500 text-[10px] truncate max-w-[200px]">{event.error?.slice(0, 80)}</span>
        {(event.attempt ?? 1) > 1 && <span className="text-[10px] text-amber-500 shrink-0">próba {event.attempt}</span>}
      </button>
      {expanded && <div className="ml-8 text-xs text-red-700 bg-red-50 rounded p-2 border border-red-100 mt-1 break-words">{event.error}</div>}
    </div>
  )
}

function EventRow({ event, index }: { event: AgentEvent; index: number }) {
  const type = event.type
  if (type === 'agent_thinking') {
    const step = (event as any).step ?? index
    return (
      <div className="flex items-center gap-2 py-1 text-xs text-slate-400 pl-2">
        <Brain className="w-3.5 h-3.5 shrink-0 text-brand-300 animate-pulse" />
        <span className="font-medium text-slate-500">Krok {step}</span>
        <span>— agent wybiera narzędzie…</span>
        <div className="ml-auto w-12 bg-slate-100 rounded-full h-1 shrink-0">
          <div className="bg-brand-300 h-1 rounded-full" style={{ width: `${Math.min(100, (step / 24) * 100)}%` }} />
        </div>
      </div>
    )
  }
  if (type === 'tool_call') return <ToolCallRow event={event} />
  if (type === 'tool_result') return <ToolResultRow event={event} />
  if (type === 'tool_error') return <ToolErrorRow event={event} />
  if (type === 'quality_report') {
    const ev = event as any
    const warnings: string[] = ev.warnings ?? []
    const avg: number | undefined = ev.stats?.avg_confidence ?? ev.avg_confidence
    const [expanded, setExpanded] = useState(false)
    return (
      <div className="py-0.5 pl-1 border-l-2 border-amber-200 ml-1">
        <button onClick={() => setExpanded(!expanded)} className="flex items-center gap-2 text-xs w-full text-left hover:bg-amber-50 rounded px-1.5 py-1">
          {expanded ? <ChevronDown className="w-3 h-3 text-amber-400 shrink-0" /> : <ChevronRight className="w-3 h-3 text-amber-400 shrink-0" />}
          <ShieldAlert className="w-3.5 h-3.5 shrink-0 text-amber-500" />
          <span className="font-medium text-amber-700">Raport jakości</span>
          {avg !== undefined && <span className={`ml-auto text-xs font-medium ${avg >= 0.75 ? 'text-green-600' : avg >= 0.60 ? 'text-amber-600' : 'text-red-600'}`}>{(avg * 100).toFixed(0)}%</span>}
        </button>
        {expanded && warnings.length > 0 && (
          <ul className="ml-8 mt-1 space-y-0.5">
            {warnings.map((w, i) => (
              <li key={i} className="flex items-start gap-1.5 text-xs text-amber-700">
                <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5 text-amber-400" />{w}
              </li>
            ))}
          </ul>
        )}
      </div>
    )
  }
  if (type === 'agent_memory') {
    return (
      <div className="py-1 pl-1 border-l-2 border-purple-200 ml-1">
        <div className="flex items-start gap-2 text-xs px-1.5 py-1">
          <BookMarked className="w-3.5 h-3.5 shrink-0 text-purple-400 mt-0.5" />
          <div>
            <span className="font-medium text-purple-700">Pamięć zapisana</span>
            <p className="text-purple-600 italic mt-0.5">{event.memory?.observation}</p>
            {event.memory?.improvement && <p className="text-purple-500 text-[10px] mt-0.5">→ {event.memory.improvement}</p>}
          </div>
        </div>
      </div>
    )
  }
  if (type === 'hint_injected') {
    return (
      <div className="flex items-center gap-2 py-1 text-xs pl-2">
        <MessageSquare className="w-3.5 h-3.5 shrink-0 text-purple-400" />
        <span className="font-medium text-purple-700">Wskazówka:</span>
        <span className="text-purple-600">{(event as any).hint}</span>
      </div>
    )
  }
  if (type === 'partial_summary') {
    const wi = (event as any).window_index ?? 0; const wc = (event as any).window_count ?? 1
    return <div className="flex items-center gap-2 py-1 text-xs pl-2 text-blue-600"><BookMarked className="w-3 h-3" />Streszczenie cząstkowe {wi + 1}/{wc} gotowe</div>
  }
  return null
}

export default function AgentLogPage() {
  const { session } = useSessionStore()
  const navigate = useNavigate()
  const bottomRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [filter, setFilter] = useState('all')

  const isProcessing = session !== null &&
    !['idle', 'done', 'error'].includes(session.processing.step)

  useEffect(() => {
    if (autoScroll && isProcessing) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [session?.agentEvents.length, autoScroll, isProcessing])

  if (!session) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-400">
        <Brain className="w-12 h-12 mb-4 opacity-30" />
        <p className="font-medium">Brak aktywnej sesji agenta.</p>
        <button onClick={() => navigate('/')} className="mt-4 text-brand-600 underline text-sm">Wróć</button>
      </div>
    )
  }

  const allEvents = session.agentEvents.filter(
    e => !['agent_start','result','segment_chunk','translation_chunk',
           'partial_segments','progress','diarization_update','segment_update'].includes(e.type)
  )
  const filtered = filter === 'all' ? allEvents
    : filter === 'errors' ? allEvents.filter(e => e.type === 'tool_error')
    : filter === 'tools' ? allEvents.filter(e => e.type === 'tool_call' || e.type === 'tool_result')
    : allEvents.filter(e => e.type === filter)

  const toolCalls = allEvents.filter(e => e.type === 'tool_call').length
  const errorCount = allEvents.filter(e => e.type === 'tool_error').length
  const thinkingCount = allEvents.filter(e => e.type === 'agent_thinking').length
  const memCount = allEvents.filter(e => e.type === 'agent_memory').length

  return (
    <div className="flex flex-col h-full bg-white">
      <div className="bg-white border-b border-slate-200 px-4 py-3 flex items-center gap-3 shrink-0">
        <button onClick={() => navigate(-1)} className="text-slate-400 hover:text-slate-700 p-1 rounded hover:bg-slate-50">
          <ArrowLeft className="w-4 h-4" />
        </button>
        <Brain className="w-4 h-4 text-brand-600" />
        <div className="flex-1 min-w-0">
          <h1 className="font-semibold text-slate-900 text-sm">Log agenta</h1>
          <p className="text-[11px] text-slate-400 truncate">{session.fileName}</p>
        </div>
        {isProcessing && (
          <div className="flex items-center gap-2 shrink-0">
            <span className="flex items-center gap-1.5 text-xs text-brand-600 bg-brand-50 px-2 py-1 rounded-full">
              <span className="w-1.5 h-1.5 bg-brand-500 rounded-full animate-pulse" />Na żywo
            </span>
            <button onClick={() => setAutoScroll(v => !v)}
              className={`text-[10px] px-2 py-0.5 rounded border ${autoScroll ? 'border-brand-200 text-brand-600' : 'border-slate-200 text-slate-400'}`}>
              scroll {autoScroll ? 'ON' : 'OFF'}
            </button>
          </div>
        )}
      </div>

      <div className="border-b border-slate-100 px-4 py-1.5 flex flex-wrap gap-3 text-xs text-slate-500 bg-slate-50 shrink-0 items-center">
        <span><strong className="text-slate-700">{toolCalls}</strong> wywołań</span>
        <span><strong className="text-slate-600">{thinkingCount}</strong> kroków</span>
        {memCount > 0 && <span><strong className="text-purple-600">{memCount}</strong> wspomnień</span>}
        {errorCount > 0 && <span><strong className="text-red-500">{errorCount}</strong> błędów</span>}
        <div className="ml-auto flex gap-1">
          {[{ key: 'all', label: 'Wszystko' }, { key: 'tools', label: 'Narzędzia' }, { key: 'errors', label: 'Błędy' }, { key: 'quality_report', label: 'Jakość' }].map(({ key, label }) => (
            <button key={key} onClick={() => setFilter(key)}
              className={`text-[10px] px-2 py-0.5 rounded border transition-colors ${filter === key ? 'border-brand-300 bg-brand-50 text-brand-700' : 'border-slate-200 text-slate-500 hover:border-slate-300'}`}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {isProcessing && (
        <div className="bg-brand-50 border-b border-brand-100 px-4 py-1.5 flex items-center gap-3 shrink-0">
          <div className="flex-1 bg-brand-100 rounded-full h-1.5">
            <div className="bg-brand-500 h-1.5 rounded-full transition-all duration-500" style={{ width: `${session.processing.progress ?? 0}%` }} />
          </div>
          <span className="text-xs text-brand-700 shrink-0 truncate max-w-[200px]">{session.processing.message}</span>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5"
        onScroll={e => { const el = e.currentTarget; setAutoScroll(el.scrollHeight - el.scrollTop - el.clientHeight < 60) }}>
        {filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-slate-400 text-sm">
            <Brain className="w-8 h-8 mb-2 opacity-30" /><p>Brak zdarzeń.</p>
          </div>
        ) : filtered.map((event, idx) => <EventRow key={idx} event={event} index={idx} />)}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
