import { useState, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Loader2, AlertCircle, Cpu, SlidersHorizontal } from 'lucide-react'
import { useSessionStore } from '../../store/sessionStore'
import { useSettingsStore } from '../../store/settingsStore'
import { VectorStore } from '../../lib/rag/vectorStore'
import { ollamaEmbed } from '../../lib/llm/ollama'
import { computeRagEntries } from '../../hooks/useProcessingPipeline'
import { speakerDisplayName } from '../../lib/speakerUtils'
import type { Segment } from '@heimdall/shared-types'

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

// ── Polish text search helpers ─────────────────────────────────────────────

const PL_MAP: Record<string, string> = {
  ą:'a', ć:'c', ę:'e', ł:'l', ń:'n', ó:'o', ś:'s', ź:'z', ż:'z',
}

function normStr(s: string): string {
  return s.toLowerCase()
    .replace(/[ąćęłńóśźż]/g, c => PL_MAP[c] ?? c)
    .replace(/[^a-z0-9]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function tokenize(s: string): string[] {
  return normStr(s).split(' ').filter(w => w.length >= 3)
}

function levenshtein(a: string, b: string): number {
  if (a === b) return 0
  const row = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i++) {
    let prev = i - 1
    row[0] = i
    for (let j = 1; j <= b.length; j++) {
      const tmp = row[j]
      row[j] = a[i - 1] === b[j - 1] ? prev : 1 + Math.min(prev, row[j], row[j - 1])
      prev = tmp
    }
  }
  return row[b.length]
}

/** Consonant skeleton for Polish phonetic matching */
function phoneticKey(word: string): string {
  let w = normStr(word)
  // normalize phonetically equivalent clusters
  w = w.replace(/rz/g,'z').replace(/ch/g,'h').replace(/sz/g,'s')
       .replace(/cz/g,'c').replace(/dz/g,'z').replace(/dz/g,'z')
  // collapse duplicate chars
  w = w.replace(/(.)(\1)+/g, '$1')
  // remove vowels → consonant skeleton
  return w.replace(/[aeiou]/g, '')
}

type MatchKind = 'exact' | 'fuzzy' | 'phonetic'

function bestWordMatch(
  qw: string,
  segTokens: string[],
): { kind: MatchKind; sim: number } | null {
  let best: { kind: MatchKind; sim: number } | null = null
  for (const sw of segTokens) {
    // exact / prefix
    if (sw === qw || sw.startsWith(qw)) return { kind: 'exact', sim: 1 }
    // fuzzy (Levenshtein)
    if (qw.length >= 4 && sw.length >= 3) {
      const dist = levenshtein(qw, sw)
      const sim = 1 - dist / Math.max(qw.length, sw.length)
      if (sim >= 0.72 && (!best || sim > best.sim)) best = { kind: 'fuzzy', sim }
    }
    // phonetic consonant skeleton
    if (qw.length >= 4 && sw.length >= 4) {
      const qk = phoneticKey(qw)
      const sk = phoneticKey(sw)
      if (qk.length >= 2 && qk === sk && (!best || best.sim < 0.7))
        best = { kind: 'phonetic', sim: 0.7 }
    }
  }
  return best
}

function scoreTextMatch(
  queryTokens: string[],
  segText: string,
): { score: number; kind: MatchKind } | null {
  if (!queryTokens.length) return null
  const toks = tokenize(segText)
  if (!toks.length) return null
  let totalSim = 0
  let matched = 0
  let dominant: MatchKind = 'phonetic'
  for (const qw of queryTokens) {
    const m = bestWordMatch(qw, toks)
    if (m) {
      totalSim += m.sim
      matched++
      if (m.kind === 'exact') dominant = 'exact'
      else if (m.kind === 'fuzzy' && dominant !== 'exact') dominant = 'fuzzy'
    }
  }
  if (!matched) return null
  // coverage × average similarity
  return { score: (matched / queryTokens.length) * (totalSim / matched), kind: dominant }
}

// ─────────────────────────────────────────────────────────────────────────────

interface SearchHit {
  matchIdx: number
  score: number
  method: 'text' | 'semantic'
  matchKind?: MatchKind
  /** Full chunk text returned by the vector store (semantic only) */
  chunkText?: string
  chunkSegmentIds?: number[]
  chunkStart?: number
  chunkEnd?: number
}

/** Render a window of segments around a hit, highlighting the match */
function SegmentWindow({
  segments,
  matchIdx,
  context = 3,
  speakerProfiles = {},
}: {
  segments: Segment[]
  matchIdx: number
  context?: number
  speakerProfiles?: Record<string, import('@heimdall/shared-types').SpeakerProfile>
}) {
  const start = Math.max(0, matchIdx - context)
  const end = Math.min(segments.length - 1, matchIdx + context)

  return (
    <div className="space-y-0.5">
      {segments.slice(start, end + 1).map((seg, relIdx) => {
        const isMatch = start + relIdx === matchIdx
        const hasTranslation = seg.text_pl && seg.text_pl !== seg.text
        return (
          <div
            key={seg.id}
            className={[
              'flex gap-3 px-3 py-2 rounded-lg transition-colors',
              isMatch
                ? 'bg-brand-50 border border-brand-200'
                : 'text-slate-500',
            ].join(' ')}
          >
            <span className="text-xs text-slate-400 w-16 shrink-0 pt-0.5 tabular-nums">
              {fmtTime(seg.start)}
            </span>
            <div className="flex-1 min-w-0">
              <span
                className={`text-xs font-semibold mr-2 ${SPEAKER_COLORS[seg.speaker] ?? 'text-slate-600'}`}
              >
                {speakerDisplayName(seg.speaker, speakerProfiles)}
              </span>
              {hasTranslation && (
                <span
                  className={`text-sm leading-snug block ${isMatch ? 'text-slate-900' : 'text-slate-500'}`}
                >
                  {seg.text_pl}
                </span>
              )}
              <span
                className={[
                  'text-sm leading-snug',
                  hasTranslation
                    ? `text-xs italic ${isMatch ? 'text-slate-500' : 'text-slate-400'}`
                    : isMatch
                    ? 'text-slate-900'
                    : 'text-slate-500',
                ].join(' ')}
              >
                {hasTranslation ? `(${seg.text})` : seg.text}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

type SearchMethod = 'text' | 'semantic'

export default function SearchPage() {
  const { session, setRagEntries } = useSessionStore()
  const { settings } = useSettingsStore()
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [method, setMethod] = useState<SearchMethod>('text')
  const [hits, setHits] = useState<SearchHit[]>([])
  const [loading, setLoading] = useState(false)
  const [buildingRag, setBuildingRag] = useState(false)
  const [ragError, setRagError] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [searched, setSearched] = useState(false)

  // Search options
  const [showOptions, setShowOptions] = useState(false)
  const [topK, setTopK] = useState(8)
  const [minScore, setMinScore] = useState(0.28)
  const [useMMR, setUseMMR] = useState(true)
  const [mmrLambda, setMmrLambda] = useState(0.6)
  const [contextWin, setContextWin] = useState(3)

  // Derive before callbacks so hooks always run in same order
  const segments = session?.segments ?? []
  const profiles = session?.speakerProfiles ?? {}
  const hasRag = (session?.ragEntries.length ?? 0) > 0

  const buildRag = useCallback(async () => {
    if (!segments.length) return
    setBuildingRag(true)
    setRagError(null)
    try {
      const cfg = {
        baseUrl: settings.ollamaUrl,
        model: settings.ollamaModel,
        embeddingModel: settings.ollamaEmbeddingModel,
      }
      const entries = await computeRagEntries(segments, cfg)
      setRagEntries(entries)
    } catch (err: any) {
      setRagError(err?.message ?? 'Błąd budowania indeksu RAG.')
    } finally {
      setBuildingRag(false)
    }
  }, [segments, settings, setRagEntries])

  const doSearch = useCallback(async () => {
    if (!query.trim() || !session) return
    setError(null)
    setLoading(true)
    setSearched(true)

    try {
      if (method === 'text') {
        const qTokens = tokenize(query)
        if (!qTokens.length) {
          setHits([])
          setLoading(false)
          return
        }
        const results: SearchHit[] = []
        segments.forEach((seg, idx) => {
          const haystack = [seg.text, seg.text_pl].filter(Boolean).join(' ')
          const m = scoreTextMatch(qTokens, haystack)
          if (m) results.push({ matchIdx: idx, score: m.score, method: 'text', matchKind: m.kind })
        })
        results.sort((a, b) => b.score - a.score)
        setHits(results)
      } else {
        if (!hasRag) {
          setError('Indeks RAG niedostępny — kliknij "Zbuduj indeks" powyżej.')
          setHits([])
          setLoading(false)
          return
        }
        const cfg = {
          baseUrl: settings.ollamaUrl,
          model: settings.ollamaModel,
          embeddingModel: settings.ollamaEmbeddingModel,
        }
        const [queryEmbedding] = await ollamaEmbed(cfg, [query])
        const store = VectorStore.fromEntries(session.ragEntries)
        const topChunks = store.search(queryEmbedding, { topK, minScore, useMMR, mmrLambda, hybridAlpha: 0.75, queryText: query })
        const seen = new Set<number>()
        const results: SearchHit[] = []
        for (const chunk of topChunks) {
          const anchorSegId = chunk.metadata.segmentIds?.[0]
          if (anchorSegId == null) continue
          const segIdx = segments.findIndex((s) => s.id === anchorSegId)
          if (segIdx === -1 || seen.has(segIdx)) continue
          seen.add(segIdx)
          results.push({
            matchIdx: segIdx,
            score: chunk.score,
            method: 'semantic',
            chunkText: chunk.text,
            chunkSegmentIds: chunk.metadata.segmentIds,
            chunkStart: chunk.metadata.start as number | undefined,
            chunkEnd: chunk.metadata.end as number | undefined,
          })
        }
        setHits(results)
      }
    } catch (err: any) {
      setError(err?.message ?? 'Błąd wyszukiwania.')
      setHits([])
    } finally {
      setLoading(false)
    }
  }, [query, method, segments, hasRag, session, settings, topK, minScore, useMMR, mmrLambda])

  // Redirect when session is cleared — useEffect so hooks always run
  useEffect(() => {
    if (!session) navigate('/')
  }, [session, navigate])

  if (!session) return null

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-6 py-4">
        <h1 className="font-semibold text-slate-900">Szukaj w transkrypcji</h1>
        <p className="text-sm text-slate-500 mt-0.5 truncate">{session.fileName}</p>
      </div>

      {/* Search bar */}
      <div className="bg-white border-b border-slate-200 px-6 py-3 flex items-center gap-3">
        {/* Method toggle */}
        <div className="flex rounded-lg border border-slate-200 overflow-hidden shrink-0">
          {(['text', 'semantic'] as SearchMethod[]).map((m) => (
            <button
              key={m}
              onClick={() => setMethod(m)}
              className={[
                'px-3 py-1.5 text-xs font-medium transition-colors',
                method === m
                  ? 'bg-brand-600 text-white'
                  : 'text-slate-600 hover:bg-slate-50',
              ].join(' ')}
            >
              {m === 'text' ? 'Tekst' : 'Semantyczne'}
            </button>
          ))}
        </div>

        {/* Input */}
        <div className="flex-1 flex items-center gap-2 border border-slate-200 rounded-xl px-3 py-2 bg-slate-50 focus-within:border-brand-400 focus-within:bg-white transition-colors">
          <Search className="w-4 h-4 text-slate-400 shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && doSearch()}
            placeholder={
              method === 'text'
                ? 'Szukaj frazy w transkrypcji…'
                : 'Szukaj semantycznie (znaczenie)…'
            }
            className="flex-1 bg-transparent text-sm text-slate-800 placeholder-slate-400 outline-none"
          />
        </div>

        <button
          onClick={() => setShowOptions((v) => !v)}
          aria-label="Opcje wyszukiwania"
          className={[
            'flex items-center gap-1.5 px-3 py-2 rounded-xl border text-xs font-medium transition-colors shrink-0',
            showOptions
              ? 'border-brand-300 bg-brand-50 text-brand-700'
              : 'border-slate-200 text-slate-600 hover:bg-slate-50',
          ].join(' ')}
        >
          <SlidersHorizontal className="w-3.5 h-3.5" />
          Opcje
        </button>

        <button
          onClick={doSearch}
          disabled={loading || !query.trim()}
          className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-brand-600 text-white text-sm font-medium hover:bg-brand-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Search className="w-4 h-4" />
          )}
          Szukaj
        </button>
      </div>

      {/* ── Options panel ────────────────────────────────────────────── */}
      {showOptions && (
        <div className="px-6 py-4 bg-white border-b border-slate-200">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-4">
            {/* top-K */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-600">Wyniki (top-K)</label>
              <input
                type="number" min={1} max={20} value={topK}
                onChange={(e) => setTopK(Math.max(1, Math.min(20, Number(e.target.value))))}
                className="w-full border border-slate-200 rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:border-brand-400"
              />
            </div>

            {/* minScore */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-600">
                Min. podobieństwo
                <span className="ml-1 font-normal text-slate-400">{(minScore * 100).toFixed(0)}%</span>
              </label>
              <input
                type="range" min={0} max={0.9} step={0.01} value={minScore}
                onChange={(e) => setMinScore(Number(e.target.value))}
                className="w-full h-1.5 accent-brand-600 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>szeroka</span><span>ścisła</span>
              </div>
            </div>

            {/* MMR */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-slate-600">Różnorodność (MMR)</label>
                <button
                  onClick={() => setUseMMR((v) => !v)}
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                    useMMR ? 'bg-brand-600' : 'bg-slate-300'
                  }`}
                >
                  <span className={`inline-block h-4 w-4 rounded-full bg-white shadow transition-transform ${
                    useMMR ? 'translate-x-4' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
              {useMMR && (
                <>
                  <input
                    type="range" min={0} max={1} step={0.05} value={mmrLambda}
                    onChange={(e) => setMmrLambda(Number(e.target.value))}
                    className="w-full h-1.5 accent-brand-600 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-400">
                    <span>różnorodne</span>
                    <span className="text-slate-500">λ={mmrLambda.toFixed(2)}</span>
                    <span>trafne</span>
                  </div>
                </>
              )}
            </div>

            {/* Context window */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-600">
                Kontekst
                <span className="ml-1 font-normal text-slate-400">±{contextWin} seg.</span>
              </label>
              <input
                type="range" min={1} max={8} step={1} value={contextWin}
                onChange={(e) => setContextWin(Number(e.target.value))}
                className="w-full h-1.5 accent-brand-600 cursor-pointer"
              />
            </div>
          </div>
        </div>
      )}

      {/* Method hint + RAG build banner */}
      <div className="px-6 py-2 bg-slate-50 border-b border-slate-100 text-xs text-slate-500">
        {method === 'text'
          ? 'Wyszukiwanie tekstowe — normalizacja, literówki (Levenshtein), fonetyka polska.'
          : 'Wyszukiwanie semantyczne — podobieństwo znaczeniowe przez embeddingi Ollama.'}
        {hasRag && (
          <span className="ml-2 text-green-600 font-medium">✓ Indeks RAG: {session.ragEntries.length} fragmentów</span>
        )}
      </div>

      {/* RAG unavailable banner */}
      {!hasRag && segments.length > 0 && (
        <div className="mx-6 mt-4 flex items-center gap-3 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3">
          <AlertCircle className="w-5 h-5 text-amber-500 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-amber-800">Indeks RAG niedostępny</p>
            <p className="text-xs text-amber-700 mt-0.5">
              Wyszukiwanie semantyczne wymaga embeddingów Ollama (<code className="font-mono">nomic-embed-text</code>).
              Upewnij się, że Ollama działa i model jest dostępny — sprawdź URL w Ustawieniach.
            </p>
            {ragError && <p className="text-xs text-red-600 mt-1">{ragError}</p>}
          </div>
          <button
            onClick={buildRag}
            disabled={buildingRag}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-600 text-white text-xs font-medium hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shrink-0"
          >
            {buildingRag ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Cpu className="w-3.5 h-3.5" />
            )}
            {buildingRag ? 'Buduję…' : 'Zbuduj indeks'}
          </button>
        </div>
      )}

      {/* Results */}
      <div className="flex-1 overflow-auto px-6 py-4 space-y-4">
        {error && (
          <div className="flex items-start gap-3 bg-red-50 border border-red-200 rounded-xl p-4">
            <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        {searched && !loading && !error && hits.length === 0 && (
          <p className="text-slate-400 text-center py-12 text-sm">
            Brak wyników dla zapytania „{query}".
          </p>
        )}

        {!searched && segments.length === 0 && (
          <p className="text-slate-400 text-center py-12 text-sm">
            Brak transkrypcji. Wróć do strony głównej i rozpocznij przetwarzanie.
          </p>
        )}

        {hits.map((hit, i) => (
          <div key={i} className="bg-white rounded-xl border border-slate-200 overflow-hidden">
            {/* Hit header */}
            <div className="flex items-center gap-3 px-4 py-2.5 border-b border-slate-100 bg-slate-50">
              <span className="text-xs font-semibold text-slate-500">#{i + 1}</span>
              <span className="text-xs text-slate-500">
                {fmtTime(segments[hit.matchIdx]?.start ?? 0)}
              </span>
              <div className="ml-auto flex items-center gap-1.5">
                {hit.method === 'text' && hit.matchKind && (
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                    hit.matchKind === 'exact' ? 'bg-green-100 text-green-700'
                    : hit.matchKind === 'fuzzy' ? 'bg-amber-100 text-amber-700'
                    : 'bg-purple-100 text-purple-700'
                  }`}>
                    {hit.matchKind === 'exact' ? 'dokładne'
                     : hit.matchKind === 'fuzzy' ? 'literówka'
                     : 'fonetyczne'}
                  </span>
                )}
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  hit.method === 'semantic' ? 'bg-brand-100 text-brand-700' : 'bg-slate-100 text-slate-500'
                }`}>
                  {hit.method === 'semantic'
                    ? `semantyczne · ${Math.round(hit.score * 100)}%`
                    : `${Math.round(hit.score * 100)}%`}
                </span>
              </div>
            </div>

            {/* Chunk preview (semantic) */}
            {hit.chunkText && (
              <div className="mx-3 mb-2 px-3 py-2.5 rounded-lg bg-indigo-50 border border-indigo-100">
                <p className="text-[10px] font-semibold text-indigo-500 uppercase tracking-wide mb-1">
                  Dopasowany fragment RAG
                  {hit.chunkStart != null && (
                    <span className="ml-1.5 normal-case font-normal text-indigo-400">
                      {fmtTime(hit.chunkStart)}
                      {hit.chunkEnd != null && ` – ${fmtTime(hit.chunkEnd)}`}
                    </span>
                  )}
                </p>
                <p className="text-xs text-slate-700 leading-relaxed line-clamp-5">{hit.chunkText}</p>
              </div>
            )}

            {/* Context window */}
            <div className="p-3">
              <SegmentWindow segments={segments} matchIdx={hit.matchIdx} context={contextWin} speakerProfiles={profiles} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
