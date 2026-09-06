"""LLM inference backends.

A backend's only job is to turn a prompt into raw text. It never parses, never
scores, and never decides anything — so results stay re-scorable offline and
swapping models never touches strategy code.
"""

from cosq.backends.base import LLMBackend
from cosq.backends.mock import MockBackend

__all__ = ["LLMBackend", "MockBackend", "load_backend"]


def load_backend(spec: str | dict[str, object]) -> LLMBackend:
    """Build a backend from a config path, a registered name, or a config mapping."""
    from cosq.config import ModelConfig, load_yaml

    if isinstance(spec, str):
        if spec in ("mock", "scripted"):
            cfg = ModelConfig(id=spec, revision="n/a", backend=spec)
        else:
            cfg = ModelConfig.from_mapping(load_yaml(spec))
    else:
        cfg = ModelConfig.from_mapping(spec)
    return cfg.build()
