"""
Translation using Ollama (sync, runs in thread pool).
Translates speaker turns (merged consecutive same-speaker segments) for efficiency.
"""
import json
import logging
import os
from typing import Callable, List, Dict, Optional

import httpx

from .cache import LRUCache

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")
TIMEOUT_BASE = 60.0           # base seconds per batch call
TIMEOUT_PER_SEGMENT = 8.0     # extra seconds per segment in batch

_SYSTEM_PROMPT = (
    "Jesteś profesjonalnym tłumaczem. "
    "ZAWSZE tłumaczysz na język POLSKI. "
    "Nigdy nie tłumacz na angielski ani żaden inny język."
)

# LRU cache for Ollama generate calls (keyed by model+url+prompt hash)
_ollama_cache: LRUCache = LRUCache(maxsize=10, name="ollama")


def _call_ollama(prompt: str, n_segments: int = 1, model: Optional[str] = None) -> str:
    cache_key = _ollama_cache.key(OLLAMA_URL, model or OLLAMA_MODEL, prompt)
    cached = _ollama_cache.get(cache_key)
    if cached is not None:
        return cached
    timeout = TIMEOUT_BASE + TIMEOUT_PER_SEGMENT * n_segments
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model or OLLAMA_MODEL,
                    "system": _SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "format": {
                        "type": "object",
                        "properties": {
                            "tlumaczenia_pl": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["tlumaczenia_pl"],
                    },
                },
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
    on_batch_done: Optional[Callable] = None,
    model: Optional[str] = None,
) -> List[Dict]:
    """
    Adds 'text_pl' field to each chunk.
    Groups segments into batches of ~10000 chars so the LLM has enough context
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

    # ── Build batches ────────────────────────────────────────────────────────
    # Hard cap on both character count and segment count so each Ollama call
    # generates a bounded number of output tokens and completes quickly.
    BATCH_CHARS = 3000
    BATCH_MAX_SEGMENTS = 15
    batches: List[Dict] = []
    current_lines: List = []
    current_len = 0

    for idx, c in enumerate(chunks):
        txt = c.get("text", "")
        current_lines.append((idx, c.get("speaker", ""), txt))
        current_len += len(txt)
        if current_len >= BATCH_CHARS or len(current_lines) >= BATCH_MAX_SEGMENTS:
            batches.append({"lines": current_lines})
            current_lines = []
            current_len = 0

    if current_lines:
        batches.append({"lines": current_lines})

    # ── Translate each batch ──────────────────────────────────────────────────
    for batch in batches:
        lines = batch["lines"]
        n = len(lines)
        # Numbered source lines so the model knows the exact expected count
        source_block = "\n".join(
            f"{k + 1}. [{sp}] {txt}" if sp else f"{k + 1}. {txt}"
            for k, (_, sp, txt) in enumerate(lines)
        )
        prompt = (
            f"Przetłumacz poniższy dialog z języka {lang_name} na język POLSKI. "
            f"Zachowaj styl wypowiedzi. "
            f"Zwróć JSON z kluczem \"tlumaczenia_pl\": tablica dokładnie {n} polskich stringów "
            f"(jedno polskie tłumaczenie na każdą ponumerowaną linię, w tej samej kolejności).\n\n"
            f"{source_block}"
        )
        raw = _call_ollama(prompt, n_segments=n, model=model)
        translated_lines: List[str] = []
        if raw:
            try:
                data = json.loads(raw)
                translated_lines = data.get("tlumaczenia_pl", [])
                if len(translated_lines) != n:
                    logger.warning(
                        f"Ollama returned {len(translated_lines)} translations for {n} segments; "
                        "falling back to original for missing entries"
                    )
            except (json.JSONDecodeError, AttributeError) as exc:
                logger.warning(f"Ollama returned non-JSON response ({exc}); falling back to original text")

        # Align translated lines back to original indices
        batch_updates: List[Dict] = []
        for k, (idx, _sp, orig_txt) in enumerate(lines):
            tl = translated_lines[k].strip() if k < len(translated_lines) else ""
            chunks[idx]["text_pl"] = tl or orig_txt
            batch_updates.append({"idx": idx, "text_pl": chunks[idx]["text_pl"]})

        # Notify caller so UI can update incrementally
        if on_batch_done and batch_updates:
            try:
                on_batch_done(batch_updates)
            except Exception as _cb_exc:
                logger.debug("on_batch_done error (ignored): %s", _cb_exc)

    return chunks


def ollama_available() -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{OLLAMA_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False
