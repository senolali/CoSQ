"""The backend contract."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from cosq.types import Completion, GenerationParams

if TYPE_CHECKING:
    from cosq.config import ModelConfig


class LLMBackend(abc.ABC):
    """Turns a prompt into raw text.

    Implementations must return the completion **verbatim**. Stripping, truncating,
    or normalising here would destroy the audit trail that ``records.jsonl`` exists
    to preserve.
    """

    #: Model identifier — a Hugging Face repo id, or a hosted provider's model name.
    model_id: str
    #: Version identity of the served weights. For local models this is a pinned
    #: commit SHA. For hosted endpoints no such handle exists, so it is a
    #: date-stamped label and :meth:`fingerprint` reports ``pinned = "false"``.
    revision: str

    @classmethod
    def from_config(cls, config: ModelConfig) -> LLMBackend:
        """Construct this backend from a :class:`~cosq.config.ModelConfig`.

        Every backend that can be named in a config implements this, so adding one
        never requires editing the config module. Wrappers such as
        :class:`~cosq.runner.cache.CachingBackend` deliberately do not: they decorate
        an already-built backend and cannot be produced from a config alone.
        """
        raise NotImplementedError(
            f"{cls.__name__} cannot be built from a config. Implement from_config() "
            "if it is meant to be selectable by name."
        )

    @abc.abstractmethod
    def generate(self, prompt: str, params: GenerationParams) -> Completion:
        """Generate one completion for ``prompt``."""

    #: Whether the served weights are immutable. False for hosted endpoints, where
    #: the provider can change the model behind a name without notice [M15].
    pinned: bool = True

    def fingerprint(self) -> dict[str, str]:
        """Identity of the served model, recorded in the run manifest."""
        return {
            "backend": type(self).__name__,
            "model_id": self.model_id,
            "revision": self.revision,
            "pinned": "true" if self.pinned else "false",
        }
