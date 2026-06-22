"""
LLM-based speaker name identification.

Analyses the transcript via Ollama and maps generic speaker labels (GŁOS_01 …)
to real names (from introductions / vocative address) or to meaningful
gender-based identifiers (Kobieta_1, Mężczyzna_1, Dziecko_1, Osoba_1) when
no name can be found.

The function is called synchronously in the thread-pool, just like the
translation step, and falls back gracefully if Ollama is unavailable.
"""
import json
import logging
import re
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

import os
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:14b")
_TIMEOUT = 90.0

# ─────────────────────────────────────────────────────────────────────────────


def identify_speakers(
    chunks: List[Dict],
    speaker_profiles: Dict[str, dict],
) -> Dict[str, str]:
    """
    Return a mapping  GŁOS_01 → display_name  for every unique speaker.

    1. Tries to extract real names via Ollama LLM.
    2. For unresolved speakers, uses gender from speaker_profiles as fallback:
       Kobieta_N / Mężczyzna_N / Dziecko_N / Osoba_N.
    """
    unique_speakers = sorted({c["speaker"] for c in chunks if c.get("speaker")})
    if not unique_speakers:
        return {}

    # Gender hint line per speaker
    profile_hints: List[str] = []
    for sp in unique_speakers:
        p = speaker_profiles.get(sp, {})
        parts: List[str] = []
        gender = p.get("gender")
        age_group = p.get("age_group")
        if gender == "zenski":
            parts.append("kobieta")
        elif gender == "meski":
            parts.append("mężczyzna")
        elif gender == "dziecko":
            parts.append("dziecko")
        if age_group and age_group != "dziecko":
            parts.append(age_group)
        suffix = f" ({', '.join(parts)})" if parts else ""
        profile_hints.append(f"{sp}{suffix}")

    # Compact transcript — prefer Polish text if available; limit tokens
    lines: List[str] = []
    total_chars = 0
    for c in chunks:
        text = (c.get("text_pl") or c.get("text") or "").strip()
        if not text:
            continue
        line = f"[{c['speaker']}]: {text}"
        lines.append(line)
        total_chars += len(line)
        if total_chars > 5000:
            lines.append("...")
            break

    transcript_text = "\n".join(lines)

    speakers_json_template = json.dumps(
        {sp: "imię lub null" for sp in unique_speakers},
        ensure_ascii=False,
        indent=2,
    )

    prompt = (
        "Poniżej transkrypcja rozmowy z oznaczonymi mówcami i ich profilami głosowymi.\n\n"
        "Mówcy i profil głosu:\n"
        + "\n".join(profile_hints)
        + "\n\nTranskrypcja:\n"
        + transcript_text
        + "\n\n"
        "Zadanie: Zidentyfikuj imię i/lub nazwisko każdego mówcy.\n"
        "Wskazówki:\n"
        "- Bezpośrednie przedstawienie: 'Dzień dobry, Krystyna Piła' → ta osoba = Krystyna Piła\n"
        "- Zwrot w wołaczu: '[GŁOS_01]: Panie Marcinie...' → GŁOS_01 zwraca się do kogoś o imieniu Marcin "
        "(sam GŁOS_01 to inna osoba!)\n"
        "- Wzmianka o nazwisku: '[GŁOS_02]: Marcin K...' → ta osoba przedstawia się jako Marcin K.\n"
        "- Jeśli nie możesz ustalić → null\n\n"
        "Odpowiedz WYŁĄCZNIE w formacie JSON (bez markdown, bez komentarzy):\n"
        + speakers_json_template
    )

    llm_names: Dict[str, Optional[str]] = {}
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()

        # Extract the JSON object from the response
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            llm_names = json.loads(json_match.group())
            # Validate: keep only string values that look like names
            for k, v in list(llm_names.items()):
                if not isinstance(v, str) or not v.strip():
                    llm_names[k] = None
                else:
                    # Reject obvious non-names returned by some models
                    stripped = v.strip()
                    if stripped.lower() in ("imię lub null", "null", "brak", "nieznany", "unknown"):
                        llm_names[k] = None
                    else:
                        llm_names[k] = stripped
        else:
            logger.warning("Speaker identification: no JSON found in LLM response: %r", raw[:300])

    except Exception as exc:
        logger.warning("Speaker identification LLM call failed: %s", exc)

    return _build_display_names(unique_speakers, llm_names, speaker_profiles)


# ─────────────────────────────────────────────────────────────────────────────


def _build_display_names(
    unique_speakers: List[str],
    llm_names: Dict[str, Optional[str]],
    speaker_profiles: Dict[str, dict],
) -> Dict[str, str]:
    """
    Merge LLM names with gender-based fallbacks.
    Gender counters are independent so we get Kobieta_1, Mężczyzna_1, …
    """
    result: Dict[str, str] = {}
    counters: Dict[str, int] = {"zenski": 0, "meski": 0, "dziecko": 0, "other": 0}

    for sp in unique_speakers:
        name = llm_names.get(sp)
        if name:
            result[sp] = name
        else:
            gender = speaker_profiles.get(sp, {}).get("gender")
            if gender == "zenski":
                counters["zenski"] += 1
                result[sp] = f"Kobieta_{counters['zenski']}"
            elif gender == "meski":
                counters["meski"] += 1
                result[sp] = f"Mężczyzna_{counters['meski']}"
            elif gender == "dziecko":
                counters["dziecko"] += 1
                result[sp] = f"Dziecko_{counters['dziecko']}"
            else:
                counters["other"] += 1
                result[sp] = f"Osoba_{counters['other']}"

    return result
