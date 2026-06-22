"""
Translation using Ollama (sync, runs in thread pool).
Translates speaker turns (merged consecutive same-speaker segments) for efficiency.
"""
import logging
import os
from typing import List, Dict

import httpx

from .cache import LRUCache

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:14b")
TIMEOUT = 120.0  # seconds per translation request

# LRU cache for Ollama generate calls (keyed by model+url+prompt hash)
_ollama_cache: LRUCache = LRUCache(maxsize=10, name="ollama")


def _call_ollama(prompt: str) -> str:
    cache_key = _ollama_cache.key(OLLAMA_URL, OLLAMA_MODEL, prompt)
    cached = _ollama_cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return ""
    _ollama_cache.put(cache_key, result)
    return result


def translate_segments_to_polish(
    chunks: List[Dict],
    detected_language: str,
) -> List[Dict]:
    """
    Adds 'text_pl' field to each chunk.
    Groups segments into batches of ~2000 chars so the LLM has enough context
    to produce coherent translations across turn boundaries.
    Chunks that are already Polish get text_pl = text.
    """
    if detected_language == "pl":
        for chunk in chunks:
            chunk["text_pl"] = chunk["text"]
        return chunks

    lang_names = {
        "en": "angielski", "ru": "rosyjski", "uk": "ukraiński",
        "de": "niemiecki", "fr": "francuski", "es": "hiszpański",
    }
    lang_name = lang_names.get(detected_language, detected_language)

    # ── Build batches of ~2000 chars ─────────────────────────────────────────
    BATCH_CHARS = 2000
    batches: List[Dict] = []   # {start_idx, end_idx, lines: [(idx, speaker, text)]}
    current_lines: List = []
    current_len = 0

    for idx, c in enumerate(chunks):
        txt = c.get("text", "")
        current_lines.append((idx, c.get("speaker", ""), txt))
        current_len += len(txt)
        if current_len >= BATCH_CHARS:
            batches.append({"lines": current_lines})
            current_lines = []
            current_len = 0

    if current_lines:
        batches.append({"lines": current_lines})

    # ── Translate each batch ──────────────────────────────────────────────────
    for batch in batches:
        lines = batch["lines"]
        # Build numbered source text preserving speaker turns
        source_block = "\n".join(
            f"[{sp}] {txt}" if sp else txt
            for (_, sp, txt) in lines
        )
        prompt = (
            f"Przetłumacz poniższy dialog z języka {lang_name} na język polski. "
            f"Zachowaj styl wypowiedzi i oznaczenia mówców w nawiasach kwadratowych. "
            f"Zwróć TYLKO tłumaczenie linijka po linijce, bez dodatkowych komentarzy.\n\n"
            f"{source_block}\n\nTłumaczenie:"
        )
        translated_block = _call_ollama(prompt)
        translated_lines = translated_block.splitlines() if translated_block else []

        # Align translated lines back to original indices (best-effort by line count)
        for k, (idx, _sp, orig_txt) in enumerate(lines):
            if k < len(translated_lines):
                # Strip speaker tag if present
                tl = translated_lines[k].strip()
                if tl.startswith("[") and "]" in tl:
                    tl = tl[tl.index("]") + 1:].strip()
                chunks[idx]["text_pl"] = tl or orig_txt
            else:
                chunks[idx]["text_pl"] = orig_txt

    return chunks


def ollama_available() -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{OLLAMA_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False
