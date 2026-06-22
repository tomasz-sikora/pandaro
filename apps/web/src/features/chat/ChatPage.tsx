import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Send, Loader2, AlertCircle, Bot, User, X, Clock, ExternalLink } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useSessionStore } from '../../store/sessionStore'
import { useSettingsStore } from '../../store/settingsStore'
import { VectorStore } from '../../lib/rag/vectorStore'
import type { SearchResult } from '../../lib/rag/vectorStore'
import { ollamaChat, ollamaEmbed } from '../../lib/llm/ollama'
import { ragQueryPrompt } from '../../lib/llm/prompts'
import { speakerDisplayName } from '../../lib/speakerUtils'
import type { Segment } from '@heimdall/shared-types'

function fmtTime(s: number) {
  const m = Math.floor(s / 60)
  const ss = Math.floor(s % 60)
  return `${m}:${ss.toString().padStart(2, '0')}`
}

const SPEAKER_COLORS: Record<string, string> = {
  GŁOS_01: 'text-blue-600', GŁOS_02: 'text-violet-600',
  GŁOS_03: 'text-amber-600', GŁOS_04: 'text-green-600',
  GŁOS_05: 'text-rose-600',  GŁOS_06: 'text-cyan-600',
}

/** Side panel showing transcript context around a timestamp */
function TranscriptPreview({
  segments,
  anchorTime,
  onClose,
  speakerProfiles = {},
}: {
  segments: Segment[]
  anchorTime: number
  onClose: () => void
  speakerProfiles?: Record<string, import('@heimdall/shared-types').SpeakerProfile>
}) {
  const anchorRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    anchorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [anchorTime])

  // Find index of segment closest to anchorTime
  const anchorIdx = segments.reduce((best, seg, i) => {
    const d = Math.abs(seg.start - anchorTime)
    return d < Math.abs(segments[best].start - anchorTime) ? i : best
  }, 0)

  // Show ±8 segments around the anchor
  const start = Math.max(0, anchorIdx - 8)
  const end = Math.min(segments.length, anchorIdx + 8)
  const visible = segments.slice(start, end)

  return (
    <div className="w-80 shrink-0 flex flex-col border-l border-slate-200 bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
        <div className="flex items-center gap-2 text-sm font-medium text-slate-700">
          <Clock className="w-4 h-4 text-brand-500" />
          Transkrypcja @ {fmtTime(anchorTime)}
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Segments */}
      <div className="flex-1 overflow-auto py-2">
        {visible.map((seg) => {
          const isAnchor = seg.id === segments[anchorIdx].id
          const hasTranslation = seg.text_pl && seg.text_pl !== seg.text
          return (
            <div
              key={seg.id}
              ref={isAnchor ? anchorRef : undefined}
              className={[
                'px-4 py-2 transition-colors',
                isAnchor ? 'bg-brand-50 border-l-2 border-brand-500' : 'hover:bg-slate-50',
              ].join(' ')}
            >
              <div className="flex items-center gap-1.5 mb-0.5">
                <span className="text-xs text-slate-400 tabular-nums">{fmtTime(seg.start)}</span>
                <span className={`text-xs font-semibold ${SPEAKER_COLORS[seg.speaker] ?? 'text-slate-500'}`}>
                  {speakerDisplayName(seg.speaker, speakerProfiles)}
                </span>
              </div>
              {/* Polish translation primary */}
              {hasTranslation && (
                <p className="text-sm text-slate-800 leading-snug">{seg.text_pl}</p>
              )}
              <p className={hasTranslation ? 'text-xs text-slate-400 italic' : 'text-sm text-slate-800 leading-snug'}>
                {seg.text}
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function ChatPage() {
  const { session, addChatMessage, updateLastAssistantMessage } = useSessionStore()
  const { settings } = useSettingsStore()
  const navigate = useNavigate()
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [previewTime, setPreviewTime] = useState<number | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!session) navigate('/')
  }, [session, navigate])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [session?.chat])

  const hasRag = (session?.ragEntries.length ?? 0) > 0
  const cfg = {
    baseUrl: settings.ollamaUrl,
    model: settings.ollamaModel,
    embeddingModel: settings.ollamaEmbeddingModel,
  }

  const send = useCallback(async () => {
    if (!input.trim() || loading || !session) return

    const question = input.trim()
    setInput('')
    setLoading(true)

    addChatMessage({
      id: crypto.randomUUID(),
      role: 'user',
      content: question,
      createdAt: Date.now(),
    })

    addChatMessage({
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      createdAt: Date.now(),
    })

    const ctrl = new AbortController()
    abortRef.current = ctrl

    try {
      // RAG retrieval
      let context = ''
      let sources: Array<{ text: string; score: number; start?: number; end?: number }> = []

      if (hasRag) {
        let queryEmbedding: number[]
        if (settings.useOllamaEmbeddings) {
          const [emb] = await ollamaEmbed(cfg, [question])
          queryEmbedding = emb
        } else {
          queryEmbedding = session.ragEntries[0]?.embedding.map(() => 0) ?? []
        }

        const store = VectorStore.fromEntries(session.ragEntries)
        const results = store.search(queryEmbedding, {
          topK: 6,
          minScore: 0.15,
          useMMR: true,
          mmrLambda: 0.6,
          hybridAlpha: 0.75,
          queryText: question,
        })
        sources = results.map((r: SearchResult) => ({
          text: r.text,
          score: r.score,
          start: r.metadata.start,
          end: r.metadata.end,
        }))
        context = results.map((r, i) => `[${i + 1}] ${r.text}`).join('\n\n')
      } else {
        // fallback: use full transcript truncated
        context = session.segments
          .map((s) => s.text)
          .join(' ')
          .slice(0, 3000)
      }

      const prompt = ragQueryPrompt(context, question)
      let answer = ''

      for await (const chunk of ollamaChat(
        cfg,
        [
          {
            role: 'system',
            content: 'Jesteś asystentem analizującym nagrania. Odpowiadaj po polsku, chyba że pytanie jest w innym języku.',
          },
          { role: 'user', content: prompt },
        ],
        ctrl.signal,
      )) {
        answer += chunk
        updateLastAssistantMessage(answer, sources)
      }

      updateLastAssistantMessage(answer, sources)
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        updateLastAssistantMessage(
          `Błąd: ${err?.message ?? 'Nie można połączyć z Ollama. Sprawdź ustawienia.'}`,
        )
      }
    } finally {
      setLoading(false)
    }
  }, [input, loading, session, hasRag, settings, cfg, addChatMessage, updateLastAssistantMessage])

  if (!session) return null

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Main chat column ──────────────────────────────────────────── */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <div className="bg-white border-b border-slate-200 px-6 py-4">
          <h1 className="font-semibold text-slate-900">Rozmowa z AI</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            {hasRag
              ? `Indeks RAG: ${session.ragEntries.length} fragmentów`
              : 'RAG niedostępny – pełna transkrypcja jako kontekst'}
          </p>
        </div>

        {/* Ollama warning */}
        {!settings.ollamaUrl && (
          <div className="mx-4 mt-3 flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm">
            <AlertCircle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
            <span className="text-amber-800">
              Skonfiguruj URL Ollamy w{' '}
              <a href="/settings" className="underline font-medium">Ustawieniach</a>.
            </span>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-auto px-4 py-4 space-y-4">
          {session.chat.length === 0 && (
            <div className="text-center text-slate-400 text-sm py-12">
              <Bot className="w-10 h-10 mx-auto mb-3 text-slate-200" />
              <p>Zadaj pytanie dotyczące nagrania.</p>
              <p className="mt-1 text-xs">Np. „O czym mówiono?" lub „Jakie były ustalenia?"</p>
            </div>
          )}

          {session.chat.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="w-7 h-7 rounded-full bg-brand-100 flex items-center justify-center shrink-0 mt-0.5">
                  <Bot className="w-4 h-4 text-brand-600" />
                </div>
              )}
              <div
                className={[
                  'max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed',
                  msg.role === 'user'
                    ? 'bg-brand-600 text-white rounded-br-sm'
                    : 'bg-white border border-slate-200 text-slate-800 rounded-bl-sm',
                ].join(' ')}
              >
                {msg.content ? (
                  msg.role === 'assistant' ? (
                    <div className="prose prose-sm prose-slate max-w-none [&_a]:text-brand-600 [&_a]:underline [&_code]:bg-slate-100 [&_code]:px-1 [&_code]:rounded [&_pre]:bg-slate-100 [&_pre]:p-2 [&_pre]:rounded">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <span>{msg.content}</span>
                  )
                ) : (
                  <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
                )}

                {/* Sources — clickable timestamps open transcript preview */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-slate-100 space-y-1">
                    <p className="text-xs text-slate-400 font-medium uppercase tracking-wide">Źródła:</p>
                    {msg.sources.map((src, i) => (
                      <button
                        key={i}
                        onClick={() => src.start != null && setPreviewTime(src.start)}
                        className={[
                          'flex items-start gap-2 w-full text-left rounded-lg hover:bg-brand-50 px-2 py-1 -mx-2 transition-colors group',
                          src.start != null ? 'cursor-pointer' : 'cursor-default',
                        ].join(' ')}
                        title={src.start != null ? `Otwórz @ ${fmtTime(src.start)}` : undefined}
                      >
                        <span className="shrink-0 text-xs font-mono bg-brand-100 text-brand-700 rounded px-1.5 py-0.5 font-semibold">
                          [{i + 1}]
                        </span>
                        {src.start != null && (
                          <span className="font-mono text-brand-500 shrink-0 text-xs mt-0.5">
                            {fmtTime(src.start)}
                          </span>
                        )}
                        <span className="text-xs text-slate-500 line-clamp-2 flex-1">
                          {src.text.slice(0, 120)}{src.text.length > 120 ? '…' : ''}
                        </span>
                        <span className="text-xs text-slate-400 shrink-0 mt-0.5">
                          {Math.round(src.score * 100)}%
                        </span>
                        {src.start != null && (
                          <ExternalLink className="w-3 h-3 text-brand-400 shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity" />
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              {msg.role === 'user' && (
                <div className="w-7 h-7 rounded-full bg-slate-100 flex items-center justify-center shrink-0 mt-0.5">
                  <User className="w-4 h-4 text-slate-500" />
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="bg-white border-t border-slate-200 px-4 py-3">
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && send()}
              placeholder="Zadaj pytanie o nagranie…"
              disabled={loading}
              className="flex-1 px-4 py-2.5 rounded-xl border border-slate-200 text-sm outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100 disabled:bg-slate-50 transition-colors"
            />
            <button
              onClick={send}
              disabled={loading || !input.trim()}
              className="p-2.5 rounded-xl bg-brand-600 text-white hover:bg-brand-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* ── Transcript preview panel ──────────────────────────────────── */}
      {previewTime !== null && session.segments.length > 0 && (
        <TranscriptPreview
          segments={session.segments}
          anchorTime={previewTime}
          onClose={() => setPreviewTime(null)}
          speakerProfiles={session.speakerProfiles}
        />
      )}
    </div>
  )
}
