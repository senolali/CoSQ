"""Outcome-rate metrics.

HR is never reported alone. Because ``correct + wrong + idk + unparseable = N``,
every abstention mechanically lowers HR, so a system that abstained on everything
would score ``HR = 0`` [M2]. :mod:`cosq.eval.selective` carries the measures that
abstention volume cannot buy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from cosq.types import OUTCOMES, ScoredRecord


@dataclass(frozen=True, slots=True)
class Counts:
    correct: int = 0
    wrong: int = 0
    idk: int = 0
    unparseable: int = 0

    @property
    def total(self) -> int:
        return self.correct + self.wrong + self.idk + self.unparseable


@dataclass(frozen=True, slots=True)
class MetricSet:
    """Per-condition rates. ``n`` is the number of scored items."""

    n: int
    counts: Counts
    hallucination_rate: float
    abstention_rate: float
    accuracy: float
    abstention_aware_accuracy: float
    effective_reliability: float
    unparseable_rate: float
    coverage: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["counts"] = asdict(self.counts)
        return payload


def count_outcomes(records: Iterable[ScoredRecord]) -> Counts:
    counter = Counter(record.outcome for record in records)
    unknown = set(counter) - set(OUTCOMES)
    if unknown:
        raise ValueError(f"unknown outcomes: {sorted(unknown)}")
    return Counts(**{outcome: counter.get(outcome, 0) for outcome in OUTCOMES})


def compute_metrics(records: Iterable[ScoredRecord]) -> MetricSet:
    """Compute every reported rate for one condition.

    ``effective_reliability`` is ``(correct - wrong) / N`` (Whitehead et al., 2022):
    abstaining scores exactly 0 and a wrong answer costs what a right one gains, so
    unlike HR it cannot be improved by abstaining more [M2].
    """
    counts = count_outcomes(records)
    n = counts.total
    if n == 0:
        raise ValueError("cannot compute metrics over zero records")

    answered = counts.correct + counts.wrong
    return MetricSet(
        n=n,
        counts=counts,
        hallucination_rate=counts.wrong / n,
        abstention_rate=counts.idk / n,
        accuracy=counts.correct / n,
        abstention_aware_accuracy=(counts.correct / answered) if answered else float("nan"),
        effective_reliability=(counts.correct - counts.wrong) / n,
        unparseable_rate=counts.unparseable / n,
        # Coverage is the fraction of parseable committed answers. Unparseable
        # outputs remain a separate measurement-failure category.
        coverage=answered / n,
    )
