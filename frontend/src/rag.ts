// Klient-side hybrydowy indeks RAG (efemeryczny, w przeglądarce).
//
// Odzwierciedla projekt z planu (SQLite-WASM FTS5 BM25 + kolumny fonetyczne +
// wektory gęste, fuzja RRF), ale jest zaimplementowany jako lekki indeks w
// pamięci w czystym TypeScript — brak zależności WASM, pełna efemeryczność i
// odporność na różnice transliteracji/błędy ASR dla języków słowiańskich.

import { normalizeText, phoneticText } from "./text";
import type { RagChunk } from "./types";

interface IndexedDoc {
  id: number;
  chunk: RagChunk;
  lexTokens: string[];
  phonTokens: string[];
}

function rrf(lists: number[][], k = 60, weights?: number[]): [number, number][] {
  const w = weights ?? lists.map(() => 1);
  const scores = new Map<number, number>();
  lists.forEach((lst, li) => {
    lst.forEach((docId, rank) => {
      scores.set(docId, (scores.get(docId) ?? 0) + w[li] * (1 / (k + rank + 1)));
    });
  });
  return [...scores.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
}

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

export interface SearchHit {
  chunk: RagChunk;
  score: number;
}

export class RagIndex {
  private docs: IndexedDoc[] = [];
  private df = new Map<string, number>();
  private avgLen = 0;

  build(chunks: RagChunk[]): void {
    this.docs = chunks.map((chunk) => {
      const lexTokens = (chunk.normalized || normalizeText(chunk.text)).split(/\s+/).filter(Boolean);
      const phonTokens = (chunk.phonetic || phoneticText(chunk.text)).split(/\s+/).filter(Boolean);
      return { id: chunk.id, chunk, lexTokens, phonTokens };
    });
    this.df.clear();
    let total = 0;
    for (const d of this.docs) {
      total += d.lexTokens.length;
      for (const t of new Set(d.lexTokens)) this.df.set(t, (this.df.get(t) ?? 0) + 1);
    }
    this.avgLen = this.docs.length ? total / this.docs.length : 0;
  }

  get size(): number {
    return this.docs.length;
  }

  clear(): void {
    this.docs = [];
    this.df.clear();
    this.avgLen = 0;
  }

  // BM25 ranking nad wybranym polem tokenów (leksykalne lub fonetyczne).
  private bm25(queryTokens: string[], field: "lexTokens" | "phonTokens"): number[] {
    const k1 = 1.5;
    const b = 0.75;
    const N = this.docs.length || 1;
    const scored: [number, number][] = [];
    for (const d of this.docs) {
      const tf = new Map<string, number>();
      for (const t of d[field]) tf.set(t, (tf.get(t) ?? 0) + 1);
      const len = d[field].length || 1;
      let score = 0;
      for (const q of queryTokens) {
        const f = tf.get(q);
        if (!f) continue;
        const df = this.df.get(q) ?? 0.5;
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * (len / (this.avgLen || 1)))));
      }
      if (score > 0) scored.push([d.id, score]);
    }
    scored.sort((a, b2) => b2[1] - a[1]);
    return scored.map(([id]) => id);
  }

  private denseRank(queryEmbedding: number[]): number[] {
    const sims: [number, number][] = [];
    for (const d of this.docs) {
      if (d.chunk.embedding && d.chunk.embedding.length) {
        sims.push([d.id, cosine(queryEmbedding, d.chunk.embedding)]);
      }
    }
    sims.sort((a, b) => b[1] - a[1]);
    return sims.map(([id]) => id);
  }

  // Wyszukiwanie hybrydowe: gęste + BM25 + fonetyczne, łączone przez RRF.
  search(query: string, queryEmbedding: number[] | null, topN = 6): SearchHit[] {
    if (!this.docs.length) return [];
    const lexQ = normalizeText(query).split(/\s+/).filter(Boolean);
    const phonQ = phoneticText(query).split(/\s+/).filter(Boolean);

    const lists: number[][] = [];
    const weights: number[] = [];
    if (queryEmbedding) {
      lists.push(this.denseRank(queryEmbedding));
      weights.push(1.0);
    }
    lists.push(this.bm25(lexQ, "lexTokens"));
    weights.push(0.8);
    lists.push(this.bm25(phonQ, "phonTokens"));
    weights.push(0.5);

    const byId = new Map(this.docs.map((d) => [d.id, d.chunk]));
    return rrf(lists, 60, weights)
      .slice(0, topN)
      .map(([id, score]) => ({ chunk: byId.get(id)!, score }))
      .filter((h) => h.chunk);
  }
}
