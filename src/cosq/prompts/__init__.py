"""Versioned prompt templates.

Prompts live here as files, never inlined in Python, because a prompt change
changes the identity of every run made with it: :func:`prompt_digest` feeds the
config hash, so an edited template produces a different ``run_id``.

Naming: ``<lang>/<name>.v<n>.txt``. Templates are formatted with ``str.format``,
so literal braces in a template must be doubled.
"""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path

_ROOT = Path(__file__).parent
DEFAULT_LANG = "en"


@cache
def load(name: str, version: int = 1, lang: str = DEFAULT_LANG) -> str:
    """Return the raw template text for ``name`` at ``version``."""
    path = _ROOT / lang / f"{name}.v{version}.txt"
    if not path.is_file():
        known = ", ".join(sorted(p.name for p in (_ROOT / lang).glob("*.txt")))
        raise FileNotFoundError(f"no prompt {path.name!r} in {lang}/; available: {known}")
    return path.read_text(encoding="utf-8")


def render(name: str, version: int = 1, lang: str = DEFAULT_LANG, **fields: object) -> str:
    """Load a template and substitute ``fields``."""
    return load(name, version, lang).format(**fields)


def prompt_digest(lang: str = DEFAULT_LANG) -> str:
    """SHA-256 over every template in ``lang``, for the config hash."""
    h = hashlib.sha256()
    for path in sorted((_ROOT / lang).glob("*.txt")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()
