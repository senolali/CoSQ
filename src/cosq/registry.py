"""A tiny plugin registry.

Backends, strategies, decision rules, datasets, and metrics are resolved by name so
that adding one never requires editing another subsystem. Registration happens as a
side effect of importing the package that defines it — see ``cosq/__init__.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

_REGISTRY: dict[str, dict[str, type]] = {}

T = TypeVar("T", bound=type)


def register(kind: str, name: str) -> Callable[[T], T]:
    """Class decorator: record ``cls`` under ``kind``/``name``."""

    def wrap(cls: T) -> T:
        bucket = _REGISTRY.setdefault(kind, {})
        if name in bucket and bucket[name] is not cls:
            raise ValueError(f"{kind} {name!r} is already registered to {bucket[name]!r}")
        bucket[name] = cls
        return cls

    return wrap


def resolve(kind: str, name: str) -> type:
    """Look up a registered class, with a helpful error listing the alternatives."""
    bucket = _REGISTRY.get(kind, {})
    if name not in bucket:
        known = ", ".join(sorted(bucket)) or "(none registered)"
        raise KeyError(f"unknown {kind} {name!r}; available: {known}")
    return bucket[name]


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))


def clear(kind: str | None = None) -> None:
    """Test helper: drop registrations."""
    if kind is None:
        _REGISTRY.clear()
    else:
        _REGISTRY.pop(kind, None)
