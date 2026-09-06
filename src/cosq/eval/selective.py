"""Selective-prediction analysis — the part HR cannot fake.

HR is satisfiable by abstention volume alone, so the mechanism claim (H2) needs
measures that hold coverage fixed: a risk–coverage curve, its area (AURC), and a
coverage-matched comparison against the baseline [M2]. See El-Yaniv & Wiener (2010),
Geifman & El-Yaniv (2017), Kamath et al. (2020).
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from cosq.types import ScoredRecord


@dataclass(frozen=True, slots=True)
class SelectivePoint:
    """One (coverage, risk) pair.

    ``coverage`` is the fraction of questions answered; ``risk`` is the error rate
    *among answered questions* — the quantity a reader actually cares about when a
    system is allowed to decline.
    """

    coverage: float
    risk: float
    n: int
    n_answered: int
    label: str = ""


def selective_point(records: Iterable[ScoredRecord], label: str = "") -> SelectivePoint:
    """Collapse one condition to its (coverage, risk) point."""
    items = list(records)
    if not items:
        raise ValueError("cannot compute a selective point over zero records")
    answered = [record for record in items if record.answered]
    wrong = sum(1 for record in answered if record.outcome != "correct")
    n = len(items)
    return SelectivePoint(
        coverage=len(answered) / n,
        risk=(wrong / len(answered)) if answered else 0.0,
        n=n,
        n_answered=len(answered),
        label=label,
    )


def risk_coverage_curve(points: Iterable[SelectivePoint]) -> list[SelectivePoint]:
    """Sort points by coverage — e.g. the τ sweep, which traces out the curve."""
    return sorted(points, key=lambda point: (point.coverage, point.risk))


def aurc(points: Sequence[SelectivePoint]) -> float:
    """Area under the risk–coverage curve by the trapezoid rule. Lower is better.

    Needs at least two distinct coverage values; a single operating point does not
    define a curve, and returning a number anyway would invite comparing points that
    are not comparable.
    """
    curve = risk_coverage_curve(points)
    xs = [point.coverage for point in curve]
    if len({round(x, 12) for x in xs}) < 2:
        raise ValueError("AURC needs at least two distinct coverage values")
    ys = [point.risk for point in curve]
    area = sum((xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2.0 for i in range(len(curve) - 1))
    return area / (xs[-1] - xs[0])


def coverage_matched_risk(
    baseline: Iterable[ScoredRecord], selective: Iterable[ScoredRecord]
) -> SelectivePoint:
    """Risk of ``baseline`` restricted to the questions ``selective`` chose to answer.

    This is the decisive test of H2. If CoSQ's advantage disappears here, the effect
    was abstention volume rather than epistemic discrimination — which is the honest
    finding, and the one this function exists to expose.
    """
    answered_ids = {record.question_id for record in selective if record.answered}
    subset = [record for record in baseline if record.question_id in answered_ids]
    if not subset:
        raise ValueError("the selective condition answered nothing the baseline covers")
    return selective_point(subset, label="coverage-matched")


def random_abstention_risk(
    baseline: Iterable[ScoredRecord],
    coverage: float,
    *,
    seed: int = 1002,
    n_boot: int = 1000,
) -> float:
    """Mean risk of the baseline when it abstains *at random* down to ``coverage``.

    The null model for "abstaining more". A selective method must beat this to have
    demonstrated anything beyond answering fewer questions.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    items = list(baseline)
    if not items:
        raise ValueError("cannot bootstrap over zero records")

    rng = random.Random(seed)
    keep = max(1, round(coverage * len(items)))
    total = 0.0
    for _ in range(n_boot):
        sample = rng.sample(items, keep)
        answered = [record for record in sample if record.answered]
        if not answered:
            continue
        total += sum(1 for record in answered if record.outcome != "correct") / len(answered)
    return total / n_boot
