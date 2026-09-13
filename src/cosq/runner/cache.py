"""A content-addressed completion cache.

The key is a hash of everything that determines the output — prompt, model, pinned
revision, decoding parameters, repeat index — so a re-run is free and an interrupted
run resumes exactly where it stopped. Nothing about the *order* of queries enters the
key, which is what makes resumption safe.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from cosq.backends.base import LLMBackend
from cosq.types import Completion, GenerationParams


def cache_key(
    prompt: str, model_id: str, revision: str, params: GenerationParams, repeat: int
) -> str:
    payload = json.dumps(
        {
            "prompt": prompt,
            "model_id": model_id,
            "revision": revision,
            "params": params.cache_key(),
            "repeat": repeat,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CompletionCache:
    """SQLite-backed store. Safe to open concurrently; writes are single-row."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self.hits = 0
        self.misses = 0
        if self.path is None:
            self._conn: sqlite3.Connection | None = None
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.RLock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS completions ("
            "  key TEXT PRIMARY KEY,"
            "  payload TEXT NOT NULL,"
            "  created_at REAL NOT NULL DEFAULT (strftime('%s','now'))"
            ")"
        )
        self._conn.commit()

    def get(self, key: str) -> Completion | None:
        if self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM completions WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        data: dict[str, Any] = json.loads(row[0])
        return Completion(
            text=data["text"],
            prompt=data["prompt"],
            model_id=data["model_id"],
            revision=data["revision"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            latency_s=data["latency_s"],
            # A cached hit costs nothing: re-running a completed experiment must not
            # inflate the reported spend.
            cost_usd=0.0,
            cached=True,
        )

    def put(self, key: str, completion: Completion) -> None:
        if self._conn is None:
            return
        payload = json.dumps(
            {
                "text": completion.text,
                "prompt": completion.prompt,
                "model_id": completion.model_id,
                "revision": completion.revision,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "latency_s": completion.latency_s,
                "cost_usd": completion.cost_usd,
            }
        )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO completions (key, payload) VALUES (?, ?)", (key, payload)
            )
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> CompletionCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class CachingBackend(LLMBackend):
    """Wraps a backend so every ``generate`` call consults the cache first.

    A real :class:`~cosq.backends.base.LLMBackend`, not a look-alike, so anything
    that accepts a backend accepts a cached one — which is what lets the runner wrap
    transparently and resume mid-experiment.
    """

    def __init__(self, backend: LLMBackend, cache: CompletionCache, repeat: int = 0) -> None:
        self._backend = backend
        self._cache = cache
        self.repeat = repeat
        self.model_id = backend.model_id
        self.revision = backend.revision

    def generate(self, prompt: str, params: GenerationParams) -> Completion:
        key = cache_key(prompt, self.model_id, self.revision, params, self.repeat)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        completion = self._backend.generate(prompt, params)
        self._cache.put(key, completion)
        return completion

    def fingerprint(self) -> dict[str, str]:
        result: dict[str, str] = self._backend.fingerprint()
        return result
