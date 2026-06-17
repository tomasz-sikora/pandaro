"""Transliteration & normalization for Slavic text (Cyrillic <-> Latin).

The goal is a *canonical* form so that differently-transliterated variants of a
name collapse to the same index key, e.g.::

    "Юрий"  -> "jurij"
    "Yuri"  -> "juri"   (close; phonetic layer bridges the rest)
    "Jurij" -> "jurij"

We deliberately avoid heavyweight ICU at runtime: a deterministic table covers
Russian/Ukrainian/Belarusian Cyrillic plus common Polish diacritics. Anything
outside the table is passed through lower-cased.
"""

from __future__ import annotations

import re
import unicodedata

# Cyrillic -> Latin (a pragmatic, search-oriented romanization, not ISO-9).
_CYRILLIC: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e",
    "є": "je", "ё": "jo", "ж": "zh", "з": "z", "и": "y", "і": "i", "ї": "ji",
    "й": "j", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "c",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e",
    "ю": "ju", "я": "ja",
}

# Polish diacritics -> base latin (kept separate; we strip combining marks too).
_POLISH: dict[str, str] = {
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s",
    "ż": "z", "ź": "z",
}

_NON_WORD = re.compile(r"[^a-z0-9]+")


def _strip_combining(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def transliterate(text: str) -> str:
    """Romanize a single token/word to lowercase latin letters."""
    out: list[str] = []
    for ch in text.lower():
        if ch in _CYRILLIC:
            out.append(_CYRILLIC[ch])
        elif ch in _POLISH:
            out.append(_POLISH[ch])
        else:
            out.append(ch)
    return _strip_combining("".join(out))


def normalize_token(token: str) -> str:
    """Canonical, comparable form of one token: transliterated + stripped."""
    return _NON_WORD.sub("", transliterate(token))


def normalize_text(text: str) -> str:
    """Normalize a whole string into space-separated canonical tokens."""
    tokens = (normalize_token(t) for t in re.split(r"\s+", text.strip()))
    return " ".join(t for t in tokens if t)
