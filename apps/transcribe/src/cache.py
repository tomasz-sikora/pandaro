"""
In-memory LRU cache for expensive pipeline results.

Stores up to `maxsize` entries (default 10). Thread-safe via Lock.
Usage:
    from .cache import LRUCache
    _cache = LRUCache(maxsize=10, name="transcribe")
    key = _cache.key(audio_sha256, language, engine)
    result = _cache.get(key)
    if result is None:
        result = expensive_computation()
        _cache.put(key, result)
"""
import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe, size-bounded in-memory LRU cache."""

    def __init__(self, maxsize: int = 10, name: str = ""):
        self._maxsize = maxsize
        self._name = name or "lru"
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def key(self, *args: Any, **kwargs: Any) -> str:
        """Build a deterministic cache key from arbitrary positional/keyword args."""
        payload = json.dumps({"a": list(args), "k": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            logger.info(f"[{self._name}] cache HIT  {key[:12]}… ({len(self._data)}/{self._maxsize})")
            return self._data[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            else:
                if len(self._data) >= self._maxsize:
                    dropped, _ = self._data.popitem(last=False)
                    logger.info(f"[{self._name}] evicted   {dropped[:12]}…")
            self._data[key] = value
            logger.info(f"[{self._name}] cache STORE {key[:12]}… ({len(self._data)}/{self._maxsize})")

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def info(self) -> dict:
        with self._lock:
            return {
                "name": self._name,
                "entries": len(self._data),
                "maxsize": self._maxsize,
            }
