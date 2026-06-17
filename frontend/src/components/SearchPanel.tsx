import { useRef, useState } from "react";
import { api } from "../api";
import type { RagIndex, SearchHit } from "../rag";
import { formatTime } from "../text";
import type { Analysis } from "../types";
import { speakerColor } from "./TranscriptView";

interface Props { index: RagIndex; onSeek: (t: number) => void; analysis: Analysis; }

type Method = "text" | "semantic";
type MatchKind = "exact" | "fuzzy" | "phonetic" | "semantic";

interface Hit extends SearchHit { method: Method; matchKind?: MatchKind; }

// ── Polish fuzzy text search (inspired by Heimdall) ──────────────────────────
const PL_MAP: Record<string, string> = {
  ą:"a",ć:"c",ę:"e",ł:"l",ń:"n",ó:"o",ś:"s",ź:"z",ż:"z",
};
function normStr(s: string): string {
  return s.toLowerCase()
    .replace(/[ąćęłńóśźż]/g, (c) => PL_MAP[c] ?? c)
    .replace(/[^a-z0-9]/g, " ").replace(/\s+/g, " ").trim();
}
function tokenize(s: string): string[] {
  return normStr(s).split(" ").filter((w) => w.length >= 2);
}
function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  const row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let prev = i - 1; row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = row[j];
      row[j] = a[i-1] === b[j-1] ? prev : 1 + Math.min(prev, row[j], row[j-1]);
      prev = tmp;
    }
  }
  return row[b.length];
}
function phoneticKey(w: string): string {
  return normStr(w)
    .replace(/rz/g,"z").replace(/ch/g,"h").replace(/sz/g,"s").replace(/cz/g,"c")
    .replace(/(.)(\1)+/g,"$1").replace(/[aeiou]/g,"");
}
function textSearch(query: string, segments: Analysis["transcript"]["segments"]): Hit[] {
  const qToks = tokenize(query);
  if (!qToks.length) return [];
  const results: Hit[] = [];
  segments.forEach((seg) => {
    const toks = tokenize(seg.text);
    if (!toks.length) return;
    let total = 0, matched = 0;
    let dominant: MatchKind = "phonetic";
    for (const qw of qToks) {
      let best: { sim: number; kind: MatchKind } | null = null;
      for (const sw of toks) {
        if (sw === qw || sw.startsWith(qw)) { best = { sim: 1, kind: "exact" }; break; }
        if (qw.length >= 4 && sw.length >= 3) {
          const sim = 1 - levenshtein(qw, sw) / Math.max(qw.length, sw.length);
          if (sim >= 0.72 && (!best || sim > best.sim)) best = { sim, kind: "fuzzy" };
        }
        if (qw.length >= 4 && sw.length >= 4) {
          const qk = phoneticKey(qw), sk = phoneticKey(sw);
          if (qk.length >= 2 && qk === sk && (!best || best.sim < 0.7))
            best = { sim: 0.7, kind: "phonetic" };
        }
      }
      if (best) { total += best.sim; matched++; if (best.kind === "exact") dominant = "exact"; else if (best.kind === "fuzzy" && dominant !== "exact") dominant = "fuzzy"; }
    }
    if (!matched) return;
    const score = (matched / qToks.length) * (total / matched);
    if (score > 0.1) results.push({ chunk: { id: seg.id, text: seg.text, speaker: seg.speaker ?? null, start: seg.start, end: seg.end, confidence: seg.confidence, normalized: "", phonetic: "", embedding: null, translation: null }, score, method: "text", matchKind: dominant });
  });
  return results.sort((a, b) => b.score - a.score).slice(0, 20);
}

// ── Context window around a hit ──────────────────────────────────────────────
function ContextWindow({ analysis, start, ctxWin }: { analysis: Analysis; start: number; ctxWin: number }) {
  const segs = analysis.transcript.segments;
  const anchorIdx = segs.findIndex((s) => s.start === start);
  if (anchorIdx < 0) return null;
  const from = Math.max(0, anchorIdx - ctxWin);
  const to = Math.min(segs.length - 1, anchorIdx + ctxWin);
  return (
    <div className="hit-context">
      {segs.slice(from, to + 1).map((seg, i) => {
        const isAnchor = from + i === anchorIdx;
        const color = speakerColor(seg.speaker);
        return (
          <div key={seg.id} className={`ctx-seg ${isAnchor ? "anchor" : ""}`}>
            <span style={{ fontSize: 11, color: "var(--text-3)", minWidth: 32, fontVariantNumeric: "tabular-nums" }}>
              {formatTime(seg.start)}
            </span>
            {seg.speaker && (
              <span style={{ color, fontSize: 11, fontWeight: 700, marginRight: 4 }}>{seg.speaker}</span>
            )}
            <span style={{ fontSize: 12, color: isAnchor ? "var(--text)" : "var(--text-2)" }}>
              {seg.text}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
export function SearchPanel({ index, onSeek, analysis }: Props) {
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState<Method>("text");
  const [hits, setHits] = useState<Hit[]>([]);
  const [busy, setBusy] = useState(false);
  const [searched, setSearched] = useState(false);
  const [showOpts, setShowOpts] = useState(false);
  const [topK, setTopK] = useState(8);
  const [minScore, setMinScore] = useState(0.25);
  const [ctxWin, setCtxWin] = useState(2);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());
  const inputRef = useRef<HTMLInputElement>(null);

  const toggleExpand = (id: number) =>
    setExpandedIds((prev) => { const s = new Set(prev); s.has(id) ? s.delete(id) : s.add(id); return s; });

  const run = async () => {
    if (!query.trim()) return;
    setBusy(true); setSearched(true);
    try {
      if (method === "text") {
        setHits(textSearch(query, analysis.transcript.segments));
      } else {
        if (!index.size) { setHits([]); return; }
        let embedding: number[] | null = null;
        try { const vecs = await api.embed([query]); embedding = vecs[0] ?? null; } catch { /* noop */ }
        const res = index.search(query, embedding, topK);
        setHits(res.filter((h) => h.score >= minScore).map((h) => ({ ...h, method: "semantic", matchKind: "semantic" })));
      }
    } finally {
      setBusy(false);
    }
  };

  const BADGE: Record<string, string> = {
    exact: "match-exact", fuzzy: "match-fuzzy", phonetic: "match-fuzzy", semantic: "match-semantic",
  };
  const BADGE_LABEL: Record<string, string> = {
    exact: "dokładny", fuzzy: "rozmyty", phonetic: "fonetyczny", semantic: "semantyczny",
  };

  return (
    <div className="search-layout">
      <div className="card">
        <h2 style={{ marginBottom: 12 }}>Wyszukiwanie</h2>

        {/* Search bar + method toggle */}
        <div className="search-bar">
          <input
            ref={inputRef}
            type="text"
            placeholder={method === "text" ? "Szukaj (z fonetyką i rozmyciem…)" : "Szukaj semantycznie…"}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
          />
          <div className="method-tabs">
            <button className={`method-tab${method === "text" ? " active" : ""}`} onClick={() => setMethod("text")}>
              Tekst
            </button>
            <button className={`method-tab${method === "semantic" ? " active" : ""}`} onClick={() => setMethod("semantic")} disabled={!index.size} title={!index.size ? "Wymaga indeksu RAG" : undefined}>
              Wektory
            </button>
          </div>
          <button onClick={run} disabled={busy || !query.trim()}>
            {busy ? <span className="spinner" /> : "🔍"}
          </button>
        </div>

        {/* Options toggle */}
        <div style={{ marginBottom: 10 }}>
          <button className="ghost sm" onClick={() => setShowOpts((s) => !s)}>
            ⚙ Opcje {showOpts ? "▲" : "▼"}
          </button>
        </div>

        {showOpts && (
          <div className="opts-panel">
            <div className="opts-grid">
              {method === "semantic" && (
                <>
                  <div className="opt-field">
                    <label className="opt-label">Top-K: {topK}</label>
                    <input type="range" min={3} max={20} value={topK} onChange={(e) => setTopK(+e.target.value)} />
                  </div>
                  <div className="opt-field">
                    <label className="opt-label">Min. score: {minScore.toFixed(2)}</label>
                    <input type="range" min={0} max={1} step={0.01} value={minScore} onChange={(e) => setMinScore(+e.target.value)} />
                  </div>
                </>
              )}
              <div className="opt-field">
                <label className="opt-label">Kontekst: ±{ctxWin} segmentów</label>
                <input type="range" min={0} max={6} value={ctxWin} onChange={(e) => setCtxWin(+e.target.value)} />
              </div>
            </div>
          </div>
        )}

        {!index.size && method === "semantic" && (
          <div className="empty">⚠ Indeks RAG nie jest gotowy — uruchom fazę RAG.</div>
        )}
      </div>

      {/* Results */}
      {searched && !busy && hits.length === 0 && (
        <div className="empty" style={{ padding: "20px 0" }}>Brak wyników dla „{query}".</div>
      )}

      {hits.length > 0 && (
        <div className="hit-list" style={{ marginTop: 10 }}>
          <div className="hint" style={{ marginBottom: 6 }}>{hits.length} wyników</div>
          {hits.map((h) => {
            const expanded = expandedIds.has(h.chunk.id);
            const color = speakerColor(h.chunk.speaker);
            return (
              <div key={h.chunk.id} className="hit-card" onClick={() => onSeek(h.chunk.start)}>
                <div className="hit-header">
                  <span style={{ fontSize: 11, color: "var(--text-2)", fontVariantNumeric: "tabular-nums" }}>
                    {formatTime(h.chunk.start)}
                  </span>
                  {h.chunk.speaker && (
                    <span style={{ color, fontSize: 11, fontWeight: 700 }}>{h.chunk.speaker}</span>
                  )}
                  {h.matchKind && (
                    <span className={`match-badge ${BADGE[h.matchKind] ?? ""}`}>
                      {BADGE_LABEL[h.matchKind] ?? h.matchKind}
                    </span>
                  )}
                  <span className="hit-score-badge">{Math.round(h.score * 100)}%</span>
                  <button
                    className="ghost sm"
                    style={{ marginLeft: 4, fontSize: 11 }}
                    onClick={(e) => { e.stopPropagation(); toggleExpand(h.chunk.id); }}
                  >
                    {expanded ? "▲" : "▼"} kontekst
                  </button>
                </div>
                <div className="hit-text">{h.chunk.text}</div>
                {expanded && ctxWin > 0 && (
                  <ContextWindow analysis={analysis} start={h.chunk.start} ctxWin={ctxWin} />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
