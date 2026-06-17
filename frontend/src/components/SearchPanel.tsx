import { useState } from "react";
import { api } from "../api";
import type { RagIndex, SearchHit } from "../rag";
import { formatTime } from "../text";

interface Props {
  index: RagIndex;
  onSeek: (t: number) => void;
}

// Wyszukiwanie hybrydowe (gęste + BM25 + fonetyczne, fuzja RRF) po stronie
// przeglądarki. Odporne na warianty transliteracji i błędy ASR (słowiańskie).
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
      embedding = null; // degraduje do BM25 + fonetyki
    }
    setHits(index.search(query, embedding));
    setBusy(false);
  };

  return (
    <div className="card">
      <h2>Wyszukiwanie</h2>
      <div className="row">
        <input
          type="text"
          placeholder="Szukaj w transkrypcie (nazwiska, słowa…)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        <button onClick={run} disabled={busy || !index.size}>
          {busy ? "…" : "Szukaj"}
        </button>
      </div>
      <ul className="hits">
        {hits.map((h) => (
          <li key={h.chunk.id} onClick={() => onSeek(h.chunk.start)}>
            <span className="seg-time">{formatTime(h.chunk.start)}</span>
            {h.chunk.speaker && <span className="seg-speaker">{h.chunk.speaker}</span>}
            <span>{h.chunk.text}</span>
          </li>
        ))}
        {!hits.length && query && !busy && <p className="muted">Brak wyników.</p>}
      </ul>
    </div>
  );
}
