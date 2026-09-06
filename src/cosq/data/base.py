"""The dataset contract and seeded sampling."""

from __future__ import annotations

import abc
import hashlib
import random
from collections.abc import Sequence

from cosq.types import Question


class DatasetAdapter(abc.ABC):
    """Yields :class:`~cosq.types.Question` objects with gold answers and options."""

    name: str

    @abc.abstractmethod
    def load(self) -> list[Question]:
        """Return every available question, in a stable order."""


def sample_questions(
    questions: Sequence[Question], n: int, seed: int
) -> tuple[list[Question], str]:
    """Draw ``n`` questions reproducibly.

    Sorting by id before sampling makes the draw independent of the order the source
    happened to yield, so the same seed gives the same subset across machines and
    library versions. Returns the sample and a SHA-256 over its ids, which is written
    into the run manifest so the exact subset is recoverable later.
    """
    if n > len(questions):
        raise ValueError(f"asked for {n} questions but only {len(questions)} are available")
    ordered = sorted(questions, key=lambda question: question.id)
    sample = random.Random(seed).sample(ordered, n) if n < len(ordered) else list(ordered)
    sample.sort(key=lambda question: question.id)

    digest = hashlib.sha256()
    for question in sample:
        digest.update(question.id.encode("utf-8"))
        digest.update(b"\x00")
    return sample, digest.hexdigest()
