"""NLP phase: entity (NER) + keyword extraction.

Keywords use the dependency-free RAKE extractor by default. The LLM (Gemma via
Ollama) optionally refines entities; if Ollama is unavailable we fall back to a
regex/heuristic NER that still surfaces capitalized names, locations and dates.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from ..clients import OllamaClient
from ..logging_setup import get_logger
from ..schemas import Entity, Keyword
from ..text import extract_keywords

log = get_logger("nlp")

_CAP = re.compile(r"\b([A-ZŁŚŻŹĆŃÓĄĘ][\wąćęłńóśźż]{2,}(?:\s+[A-ZŁŚŻŹĆŃÓĄĘ][\wąćęłńóśźż]{2,})?)\b")
_DATE = re.compile(
    r"\b(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"poniedziałek|wtorek|środa|czwartek|piątek|sobota|niedziela|"
    r"styczeń|luty|marzec|kwiecień|maj|czerwiec|lipiec|sierpień|wrzesień|"
    r"październik|listopad|grudzień)\b",
    re.IGNORECASE,
)


def heuristic_entities(text: str) -> list[Entity]:
    counts: Counter[tuple[str, str]] = Counter()
    for m in _CAP.finditer(text):
        counts[(m.group(1), "MISC")] += 1
    for m in _DATE.finditer(text):
        counts[(m.group(0), "DATE")] += 1
    return [
        Entity(text=t, type=typ, count=c)
        for (t, typ), c in counts.most_common(50)
    ]


async def llm_entities(text: str, client: OllamaClient) -> list[Entity] | None:
    """Ask Gemma to extract typed entities as JSON. Returns None on failure."""
    prompt = (
        "Wyodrębnij encje z poniższego transkryptu rozmowy. "
        "Zwróć WYŁĄCZNIE poprawny JSON: lista obiektów "
        '{"text": str, "type": "PERSON|LOC|ORG|DATE|MISC"}.\n\n'
        f"Transkrypt:\n{text[:6000]}"
    )
    try:
        raw = await client.generate(prompt, temperature=0.0)
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            return None
        data = json.loads(raw[start : end + 1])
        ents = [
            Entity(text=str(d["text"]), type=str(d.get("type", "MISC")))
            for d in data
            if d.get("text")
        ]
        return ents or None
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("nlp.llm_entities_failed", error=str(exc))
        return None


async def extract_nlp(
    text: str, *, client: OllamaClient | None = None, use_llm: bool = True
) -> tuple[list[Entity], list[Keyword]]:
    keywords: list[Keyword] = extract_keywords(text)
    entities: list[Entity] | None = None
    if use_llm and client is not None:
        entities = await llm_entities(text, client)
    if entities is None:
        entities = heuristic_entities(text)
    return entities, keywords
