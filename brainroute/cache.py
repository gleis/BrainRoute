"""Small thread-safe TTL cache used to keep hot paths off the network.

Discovery, config assembly, and model-based classification are all expensive
relative to a routing decision. Without caching, every HTTP request would issue
live calls to each configured provider before it could score a single model.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Hashable


class TTLCache:
    """A bounded, thread-safe cache with per-entry expiry."""

    def __init__(self, ttl_seconds: float, max_entries: int = 256) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._lock = threading.Lock()
        self._entries: dict[Hashable, tuple[float, Any]] = {}

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < time.monotonic():
                self._entries.pop(key, None)
                return None
            return value

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            if len(self._entries) >= self.max_entries:
                self._evict_locked()
            self._entries[key] = (time.monotonic() + self.ttl_seconds, value)

    def get_or_set(self, key: Hashable, factory: Callable[[], Any]) -> Any:
        """Return the cached value, computing it via ``factory`` on a miss.

        The factory runs outside the lock so a slow network call cannot block
        readers of unrelated keys. A duplicate concurrent computation is
        acceptable and cheaper than serialising every caller.
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _evict_locked(self) -> None:
        now = time.monotonic()
        expired = [key for key, (expires_at, _) in self._entries.items() if expires_at < now]
        for key in expired:
            self._entries.pop(key, None)
        while len(self._entries) >= self.max_entries:
            oldest = min(self._entries, key=lambda key: self._entries[key][0])
            self._entries.pop(oldest, None)
