"""Pure-logic text utilities (Slavic-aware), independent of any ML stack."""

from .chunking import chunk_transcript, plan_summary_chunks
from .confidence import annotate_transcript, is_hallucination, word_confidence
from .fusion import cosine_rank, reciprocal_rank_fusion
from .keywords import extract_keywords
from .phonetic import phonetic_code, phonetic_text
from .translit import normalize_text, normalize_token, transliterate

__all__ = [
    "chunk_transcript",
    "plan_summary_chunks",
    "annotate_transcript",
    "is_hallucination",
    "word_confidence",
    "cosine_rank",
    "reciprocal_rank_fusion",
    "extract_keywords",
    "phonetic_code",
    "phonetic_text",
    "normalize_text",
    "normalize_token",
    "transliterate",
]
