"""A minimal ``.env`` loader.

``.env.example`` tells you to copy the file and fill it in, so something has to read
it. This is that something — deliberately tiny and dependency-free rather than
pulling in `python-dotenv` for twenty lines of parsing.

**The real environment always wins.** A variable already set in the shell is never
overwritten by the file, so a CI secret or an explicit ``set``/``export`` cannot be
silently clobbered by a stale ``.env`` left in the working directory.

Loading happens in :func:`cosq.cli.main` only. Importing ``cosq`` as a library never
touches the environment — a library that mutates ``os.environ`` on import is a
surprise nobody wants.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_FILENAME = ".env"


def parse_env(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines.

    Accepts a leading ``export``, ignores blank lines and ``#`` comments, and strips
    one layer of matching single or double quotes from the value. Malformed lines are
    skipped rather than raising: a broken comment should not stop a run.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(path: str | Path | None = None) -> list[str]:
    """Load ``.env`` into ``os.environ`` and return the names actually set.

    Missing file, unreadable file, or a variable already present in the environment
    are all no-ops. Returns the keys it set, so a caller can report them without
    printing any values.
    """
    target = Path(path) if path is not None else Path.cwd() / DEFAULT_FILENAME
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    applied = []
    for key, value in parse_env(text).items():
        if key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    return applied
