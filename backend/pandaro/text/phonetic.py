"""Phonetic encoding tuned for Slavic names after romanization.

Classic Soundex is English-centric and useless for Polish/Ukrainian/Russian. We
implement a small, deterministic phonetic encoder that:

1. runs on the *transliterated* form (so Cyrillic and Latin spellings converge);
2. collapses common Slavic digraph noise (sz/sh -> S, cz/ch -> C, etc.);
3. drops vowels after the first letter and squeezes repeats — a Soundex-style
   skeleton that is robust to ASR spelling errors and transliteration variance.

This is intentionally dependency-free; for production one might swap in
Beider-Morse via the ``abydos`` library behind this same function.
"""

from __future__ import annotations

import re

from .translit import transliterate

# Order matters: longer digraphs first.
_DIGRAPHS: list[tuple[str, str]] = [
    ("shch", "S"),
    ("sch", "S"),
    ("sz", "S"),
    ("sh", "S"),
    ("cz", "C"),
    ("ch", "C"),
    ("kh", "H"),
    ("zh", "Z"),
    ("rz", "Z"),
    ("dz", "C"),
    ("ph", "F"),
    ("th", "T"),
    ("ck", "K"),
]

# Single-letter sound classes (consonants mapped to a coarse class).
_CLASS: dict[str, str] = {
    "b": "B", "p": "B", "w": "F", "f": "F", "v": "F",
    "c": "C", "z": "Z", "s": "S", "j": "J", "g": "K", "k": "K", "q": "K",
    "x": "KS", "d": "T", "t": "T", "l": "L", "r": "R", "m": "M", "n": "N",
    "h": "H", "y": "J",
}

_VOWELS = set("aeiou")
_NON_ALPHA = re.compile(r"[^a-z]")


def phonetic_code(word: str) -> str:
    """Return a short phonetic skeleton for a single word."""
    s = _NON_ALPHA.sub("", transliterate(word))
    if not s:
        return ""

    for src, dst in _DIGRAPHS:
        s = s.replace(src, dst.lower())

    # First character is kept (letter or already-mapped class token).
    first = s[0]
    head = first.upper() if first.isalpha() else first

    body: list[str] = []
    for ch in s[1:]:
        if ch.isupper():  # already a digraph class token
            body.append(ch)
        elif ch in _VOWELS:
            continue  # drop interior vowels
        else:
            body.append(_CLASS.get(ch, ""))

    code = head + "".join(body)
    # Squeeze consecutive duplicates.
    squeezed: list[str] = []
    for c in code:
        if not squeezed or squeezed[-1] != c:
            squeezed.append(c)
    return "".join(squeezed)


def phonetic_text(text: str) -> str:
    """Phonetic codes for every token, space separated (for an FTS column)."""
    codes = (phonetic_code(t) for t in re.split(r"\s+", text.strip()))
    return " ".join(c for c in codes if c)
