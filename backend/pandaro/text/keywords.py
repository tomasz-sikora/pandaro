"""Lightweight, dependency-free keyword extraction (RAKE-style).

Used as a fast default and as a fallback when the LLM-based extractor is
disabled. Polish/Slavic stopwords are included. The LLM (Gemma via Ollama) can
refine these in :mod:`pandaro.pipeline.nlp`.
"""

from __future__ import annotations

import re
from collections import Counter

from ..schemas import Keyword

# A compact Polish + English + RU/UK-romanized stopword set.
_STOPWORDS: set[str] = set(
    """
    i oraz albo lub ale wiec więc bo bowiem ponieważ że ze aby żeby by to ten ta
    te tym tego tej tych ich jego jej nasz wasz mój moja moje jest są był była było
    będzie nie tak co kto gdzie kiedy jak jaki jaka jakie czy a o u w we z za na do
    od po przy dla bez pod nad przez się siebie sobie taki która który które jako
    the a an and or but of to in on for with at by is are was were be been this that
    it as i you he she we they not no yes do does did
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
    """.split()
)

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def extract_keywords(text: str, *, top_k: int = 25, min_len: int = 3) -> list[Keyword]:
    """Return scored keyphrases using a simple RAKE-like degree heuristic."""
    words = [w.lower() for w in _TOKEN.findall(text)]
    # Build candidate phrases delimited by stopwords.
    phrases: list[list[str]] = []
    current: list[str] = []
    for w in words:
        if w in _STOPWORDS or len(w) < min_len:
            if current:
                phrases.append(current)
                current = []
        else:
            current.append(w)
    if current:
        phrases.append(current)

    freq: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    for ph in phrases:
        deg = len(ph) - 1
        for w in ph:
            freq[w] += 1
            degree[w] += deg + 1

    word_score = {w: degree[w] / freq[w] for w in freq}
    phrase_scores: Counter[str] = Counter()
    for ph in phrases:
        score = sum(word_score[w] for w in ph)
        phrase_scores[" ".join(ph)] += score

    top = phrase_scores.most_common(top_k)
    max_score = top[0][1] if top else 1.0
    return [Keyword(term=t, score=round(s / max_score, 4)) for t, s in top]
