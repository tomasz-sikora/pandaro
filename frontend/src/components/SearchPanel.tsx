import { useState } from "react";
import { api } from "../api";
import type { RagIndex, SearchHit } from "../rag";
import { formatTime } from "../text";
import { speakerColor } from "./TranscriptView";

interface Props {
  index: RagIndex;
  onSeek: (t: number) => void;
}

export function SearchPanel({ index, onSeek }: Props) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    if (!query.trim() || !index.size) return;
    setBusy(true);
    let embedding: number[] | null = null;
    try {
      const vecs = await api.embed([query]);
      embedding = vecs[0] ?? null;
    } catch {
      /* degraduje do BM25 + fonetyki */
    }
    setHits(index.search(query, embedding));
    setBusy(false);
  };

  return (
    <div className="card">
      <h2>Wyszukiwanie hybrydowe</h2>
      <p className="hint" style={{ marginBottom: 10 }}>
        BM25 + wektory bge-m3 + fonetyka cyrylicy, fuzja RRF. Odporne na warianty transliteracji.
      </p>
      <div className="row" style={{ marginBottom: 12 }}>
        <input
          type="text"
          placeholder="Szukaj w transkrypcie (nazwiska, słowa kluczowe…)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        <button onClick={run} disabled={busy || !index.size}>
          {busy ? <span className="spinner" /> : "🔍 Szukaj"}
        </button>
      </div>

      {!index.size && <div className="empty">Indeks RAG nie jest gotowy — uruchom fazę RAG.</div>}

      {hits.length > 0 && (
        <ul className="hits-list">
          {hits.map((h) => (
            <li key={h.chunk.id} className="hit-item" onClick={() => onSeek(h.chunk.start)}>
              <div className="hit-header">
                <span className="seg-time">{formatTime(h.chunk.start)}</span>
                {h.chunk.speaker && (
                  <span style={{ color: speakerColor(h.chunk.speaker), fontSize: 12, fontWeight: 600 }}>
                    {h.chunk.speaker}
                  </span>
                )}
                <span className="hit-score">score: {h.score.toFixed(3)}</span>
              </div>
              <span style={{ fontSize: 13 }}>{h.chunk.text}</span>
            </li>
          ))}
        </ul>
      )}
      {hits.length === 0 && query && !busy && (
        <div className="empty">Brak wyników dla „{query}".</div>
      )}
    </div>
  );
}
