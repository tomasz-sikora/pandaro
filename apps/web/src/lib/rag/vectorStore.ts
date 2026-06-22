import type { VectorEntry } from '@heimdall/shared-types'

// ── Math helpers ─────────────────────────────────────────────────────────────

function dot(a: number[], b: number[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

function normVec(a: number[]): number {
  return Math.sqrt(dot(a, a))
}

/** L2-normalize a vector in-place, returns the same array. */
function l2Normalize(v: number[]): number[] {
  const n = normVec(v)
  if (n === 0) return v
  for (let i = 0; i < v.length; i++) v[i] /= n
  return v
}

export function cosineSim(a: number[], b: number[]): number {
  const na = normVec(a)
  const nb = normVec(b)
  if (na === 0 || nb === 0) return 0
  return dot(a, b) / (na * nb)
}

// ── Keyword scoring (BM25-lite, Polish-aware) ────────────────────────────────

const PL_MAP: Record<string, string> = {
  ą: 'a', ć: 'c', ę: 'e', ł: 'l', ń: 'n', ó: 'o', ś: 's', ź: 'z', ż: 'z',
}

function normText(s: string): string {
  return s
    .toLowerCase()
    .replace(/[ąćęłńóśźż]/g, (c) => PL_MAP[c] ?? c)
    .replace(/[^a-z0-9]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function tokenize(s: string): string[] {
  return normText(s).split(' ').filter((w) => w.length >= 3)
}

/** Levenshtein distance (capped at maxDist for speed). */
function levenshtein(a: string, b: string, maxDist = 3): number {
  if (a === b) return 0
  if (Math.abs(a.length - b.length) > maxDist) return maxDist + 1
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

/**
 * BM25-style keyword score for a query against a document text.
 * Supports exact match, prefix match, and fuzzy (edit-distance ≤ 2).
 * Returns a normalized score ∈ [0, 1].
 */
function keywordScore(queryTokens: string[], docText: string): number {
  if (!queryTokens.length) return 0
  const docToks = tokenize(docText)
  if (!docToks.length) return 0

  // BM25 parameters
  const k1 = 1.5
  const b = 0.75
  const avgDocLen = 60 // approximate average token count per chunk
  const N = docToks.length

  let totalScore = 0
  let matched = 0

  for (const qt of queryTokens) {
    let bestSim = 0

    for (const dt of docToks) {
      let sim = 0
      if (dt === qt || dt.startsWith(qt) || qt.startsWith(dt)) {
        sim = 1.0
      } else if (qt.length >= 4 && dt.length >= 3) {
        const dist = levenshtein(qt, dt, 2)
        if (dist <= 2) sim = 1 - dist / Math.max(qt.length, dt.length)
      }
      if (sim > bestSim) bestSim = sim
    }

    if (bestSim > 0.6) {
      matched++
      // BM25-style term frequency factor
      const tf = bestSim
      const bm25 = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (N / avgDocLen)))
      totalScore += bm25 * bestSim
    }
  }

  if (matched === 0) return 0

  // Normalize by max achievable score (all terms matched perfectly)
  const maxScore = queryTokens.length * ((k1 + 1) / (1 + k1 * (1 - b + b * (N / avgDocLen))))
  return Math.min(1, totalScore / maxScore) * (matched / queryTokens.length)
}

// ── MMR ──────────────────────────────────────────────────────────────────────

function mmrSelect<T extends VectorEntry & { score: number }>(
  candidates: T[],
  topK: number,
  lambda: number,
): T[] {
  const selected: T[] = []
  const pool = [...candidates]

  while (selected.length < topK && pool.length > 0) {
    let bestIdx = 0
    let bestMmr = -Infinity

    for (let i = 0; i < pool.length; i++) {
      const relevance = pool[i].score
      let maxRedundancy = 0
      for (const sel of selected) {
        const sim = cosineSim(pool[i].embedding, sel.embedding)
        if (sim > maxRedundancy) maxRedundancy = sim
      }
      const mmr = lambda * relevance - (1 - lambda) * maxRedundancy
      if (mmr > bestMmr) {
        bestMmr = mmr
        bestIdx = i
      }
    }

    selected.push(pool[bestIdx])
    pool.splice(bestIdx, 1)
  }

  return selected
}

// ── Public API ───────────────────────────────────────────────────────────────

export interface SearchOptions {
  /** Maximum number of results to return (default: 8). */
  topK?: number
  /**
   * Minimum combined score threshold (default: 0.15).
   * Lower = broader recall.
   */
  minScore?: number
  /** Apply MMR re-ranking for diversity (default: true). */
  useMMR?: boolean
  /**
   * MMR λ ∈ [0, 1]. 1.0 = pure relevance, 0.0 = pure diversity.
   * Default: 0.6
   */
  mmrLambda?: number
  /**
   * Weight of semantic (embedding) score vs keyword score.
   * hybridAlpha=1.0 → pure semantic; 0.0 → pure keyword.
   * Default: 0.75
   */
  hybridAlpha?: number
  /**
   * Plain-text query for hybrid keyword scoring.
   * If omitted, only semantic search is used.
   */
  queryText?: string
}

export interface SearchResult extends VectorEntry {
  score: number
  /** Raw cosine similarity score (before hybrid fusion). */
  semanticScore: number
  /** BM25-style keyword overlap score (0 if queryText not provided). */
  keywordScore: number
}

export class VectorStore {
  private entries: VectorEntry[] = []

  addMany(entries: VectorEntry[]): void {
    // Store with L2-normalized embeddings for faster cosine (just dot product)
    for (const e of entries) {
      this.entries.push({ ...e, embedding: l2Normalize([...e.embedding]) })
    }
  }

  /**
   * Hybrid semantic + keyword search with MMR re-ranking.
   *
   * Scoring pipeline:
   *   1. Cosine similarity between L2-normalized query and stored embeddings.
   *   2. BM25-lite keyword score (fuzzy + exact) when `queryText` is provided.
   *   3. Linear hybrid fusion: α·sem + (1−α)·kw.
   *   4. MMR re-ranking for diversity.
   */
  search(query: number[], options: SearchOptions = {}): SearchResult[] {
    const {
      topK = 8,
      minScore = 0.15,
      useMMR = true,
      mmrLambda = 0.6,
      hybridAlpha = 0.75,
      queryText,
    } = options

    // Normalize query vector
    const qNorm = l2Normalize([...query])
    const qTokens = queryText ? tokenize(queryText) : []

    const candidates: SearchResult[] = []
    for (const e of this.entries) {
      // Semantic score (dot product of already-normalized vectors = cosine)
      const sem = dot(qNorm, e.embedding)
      // Keyword score (0 when no queryText)
      const kw = qTokens.length > 0 ? keywordScore(qTokens, e.text) : 0
      // Hybrid fusion
      const combined = hybridAlpha * sem + (1 - hybridAlpha) * kw

      if (combined >= minScore) {
        candidates.push({ ...e, score: combined, semanticScore: sem, keywordScore: kw })
      }
    }

    if (candidates.length === 0) return []

    candidates.sort((a, b) => b.score - a.score)

    if (!useMMR || candidates.length <= topK) {
      return candidates.slice(0, topK)
    }

    // Widen candidate pool 3× before MMR to keep diversity meaningful
    const pool = candidates.slice(0, topK * 3)
    return mmrSelect(pool, topK, mmrLambda) as SearchResult[]
  }

  clear(): void {
    this.entries = []
  }

  get size(): number {
    return this.entries.length
  }

  static fromEntries(entries: VectorEntry[]): VectorStore {
    const s = new VectorStore()
    s.addMany(entries)   // addMany normalizes embeddings
    return s
  }
}
