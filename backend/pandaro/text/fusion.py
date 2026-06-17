"""Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Combines several ranked result lists (dense / BM25 / phonetic) into one ranking
without needing comparable scores. Mirrors the algorithm used client-side in the
WASM RAG engine so that backend tests pin the expected behaviour.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranked lists of document ids.

    Args:
        ranked_lists: each inner sequence is doc ids ordered best-first.
        k: RRF damping constant (60 is the canonical default).
        weights: optional per-list weights (defaults to 1.0 each).
        top_n: if given, truncate the fused result.

    Returns:
        ``[(doc_id, fused_score), ...]`` ordered by descending score.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    scores: dict[int, float] = {}
    for lst, w in zip(ranked_lists, weights, strict=True):
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (k + rank + 1))

    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused[:top_n] if top_n is not None else fused


def cosine_rank(query: Sequence[float], matrix: Iterable[Sequence[float]]) -> list[int]:
    """Rank rows of ``matrix`` by cosine similarity to ``query`` (best first)."""
    import math

    def cos(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    sims = [(i, cos(query, row)) for i, row in enumerate(matrix)]
    sims.sort(key=lambda kv: (-kv[1], kv[0]))
    return [i for i, _ in sims]
