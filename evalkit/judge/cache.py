"""Disk cache for judge calls.

An eval run repeats itself constantly - reruns after a code tweak, replays,
CI. Without a cache every repeat bills the same verdicts again; with one, a
rerun over unchanged inputs costs nothing and finishes in seconds. The cache
key covers everything that changes a verdict (model, messages, decoding
parameters) and nothing that doesn't (time, request ids).
"""

import hashlib
import json
from pathlib import Path


def call_key(model: str, messages: list[dict], max_tokens: int,
             temperature: float) -> str:
    """Canonical key: sha256 over the sorted-JSON of verdict-relevant fields."""
    payload = json.dumps(
        {"model": model, "messages": messages,
         "max_tokens": max_tokens, "temperature": temperature},
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class CachedClient:
    """Wraps any client with a .chat(messages, ...) method and a .model attr."""

    def __init__(self, client, cache_path: str | Path):
        self.client = client
        self.path = Path(cache_path)
        self.hits = 0
        self.misses = 0
        if self.path.exists():
            self._cache = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._cache = {}

    @property
    def model(self) -> str:
        return self.client.model

    def chat(self, messages: list[dict], max_tokens: int = 600,
             temperature: float = 0.0, **kwargs) -> str:
        key = call_key(self.client.model, messages, max_tokens, temperature)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        out = self.client.chat(messages, max_tokens=max_tokens,
                               temperature=temperature, **kwargs)
        self._cache[key] = out
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=0),
                             encoding="utf-8")
        return out
