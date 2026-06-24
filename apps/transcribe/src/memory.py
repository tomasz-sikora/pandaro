"""
Agent memory / skill storage.

Memories are persisted in /app/data/memories.json (override via AGENT_MEMORY_PATH).
Each entry records an observation + actionable improvement from a previous session.
The agent reads all memories at session start (formatted into its system prompt)
and may write new ones after processing completes.
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MEMORY_PATH = Path(os.getenv("AGENT_MEMORY_PATH", "/app/data/memories.json"))
MAX_MEMORIES = 50   # hard cap stored on disk
PROMPT_MEMORIES = 8  # how many to inject into agent system prompt


def _load_raw() -> List[Dict]:
    if not MEMORY_PATH.exists():
        return []
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Memory load failed: %s", exc)
        return []


def load_memories() -> List[Dict]:
    """Return all memories, most-recent first, capped at MAX_MEMORIES."""
    mems = _load_raw()
    mems.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return mems[:MAX_MEMORIES]


def save_memory(
    observation: str,
    improvement: str,
    tags: Optional[List[str]] = None,
) -> Dict:
    """
    Append a new memory. Deduplicates by first-80-chars of observation text.
    Returns the saved (or existing duplicate) memory dict.
    """
    mems = _load_raw()

    obs_lower = observation.lower().strip()
    for existing in mems:
        if existing.get("observation", "").lower().strip()[:80] == obs_lower[:80]:
            logger.info("Memory dedup: similar observation already recorded")
            return existing

    new_mem: Dict = {
        "id": f"mem_{int(time.time())}",
        "observation": observation,
        "improvement": improvement,
        "tags": tags or [],
        "created_at": time.time(),
        "times_applied": 0,
    }
    mems.append(new_mem)
    mems.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    mems = mems[:MAX_MEMORIES]

    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(mems, f, ensure_ascii=False, indent=2)
        logger.info("Memory saved: %s", new_mem["id"])
    except Exception as exc:
        logger.warning("Memory persist failed: %s", exc)

    return new_mem


def format_memories_for_prompt(memories: Optional[List[Dict]] = None) -> str:
    """Format the top PROMPT_MEMORIES memories as a skills block for the agent prompt."""
    if memories is None:
        memories = load_memories()
    if not memories:
        return ""
    lines = ["\nLEARNED SKILLS FROM PREVIOUS SESSIONS:"]
    for i, m in enumerate(memories[:PROMPT_MEMORIES], 1):
        lines.append(f"  {i}. Observation: {m['observation']}")
        lines.append(f"     Improvement: {m['improvement']}")
    return "\n".join(lines)


def list_memories() -> List[Dict]:
    return load_memories()


def delete_memory(memory_id: str) -> bool:
    """Delete a memory by id. Returns True if deleted."""
    mems = _load_raw()
    before = len(mems)
    mems = [m for m in mems if m.get("id") != memory_id]
    if len(mems) == before:
        return False
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(mems, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Memory delete persist failed: %s", exc)
    return True


def clear_all_memories() -> None:
    if MEMORY_PATH.exists():
        MEMORY_PATH.unlink()
