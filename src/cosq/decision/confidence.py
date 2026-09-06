"""Graded-confidence abstention.

The binary rule asks each sub-fact "certain or not?" and then takes a conjunction, so
the decision variable is the *fraction* of sub-facts marked uncertain. With the three
or four sub-facts a model typically lists, that fraction takes only four or five
distinct values — a resolution too coarse to draw a risk-coverage curve through. In
the first GPT-4o campaign τ = 0.20 and τ = 0.22 produced byte-identical outcomes for
exactly this reason [M20].

Two things are lost before the threshold is ever applied: the binary label discards
*how* certain the model is, and the conjunction discards everything except whether any
item was flagged. This module keeps both — a confidence in ``[0, 1]`` per sub-fact,
combined by an explicit aggregator into a continuous score.

**It generalises the pre-registered rule rather than replacing it.** On binary inputs
(``certain`` → 1.0, ``uncertain`` → 0.0):

* ``ConfidenceRule(aggregator="minimum", threshold=1.0)`` reproduces the strict τ = 0
  rule exactly;
* ``ConfidenceRule(aggregator="mean", threshold=1 - τ)`` reproduces ``ThresholdRule(τ)``
  exactly.

Both equivalences are asserted by tests, so the τ sweep remains a special case of this
one and nothing pre-registered is discarded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from cosq.decision.base import DecisionOutcome
from cosq.registry import register

#: Aggregators over per-sub-fact confidences.
#:
#: * ``minimum`` — the weakest link. The continuous analogue of the conjunction, and
#:   the conservative choice: one shaky sub-fact sinks the answer.
#: * ``mean`` — forgiving. Generalises the pre-registered fraction rule.
#: * ``product`` — ``P(all correct)`` under independence. Principled, and deliberately
#:   length-sensitive: needing ten facts really is riskier than needing two.
#: * ``geometric_mean`` — the product normalised for list length, for when that
#:   length sensitivity is judged an artefact of how verbose the model was [M7].
AGGREGATORS: dict[str, str] = {
    "minimum": "weakest link; continuous analogue of the conjunction",
    "mean": "average confidence; generalises the pre-registered fraction rule",
    "product": "P(all correct) under independence; length-sensitive",
    "geometric_mean": "product normalised for list length",
}


def aggregate(confidences: Sequence[float], how: str) -> float:
    """Combine per-sub-fact confidences into one score in ``[0, 1]``."""
    if how not in AGGREGATORS:
        known = ", ".join(sorted(AGGREGATORS))
        raise ValueError(f"unknown aggregator {how!r}; available: {known}")
    if not confidences:
        raise ValueError("cannot aggregate an empty confidence list")
    for value in confidences:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence {value} outside [0, 1]")

    if how == "minimum":
        return min(confidences)
    if how == "mean":
        return sum(confidences) / len(confidences)
    if how == "product":
        return math.prod(confidences)
    product = math.prod(confidences)
    return product ** (1.0 / len(confidences))


@register("decision", "confidence")
class ConfidenceRule:
    """Abstain when the aggregate confidence falls below ``threshold``.

    Deliberately *not* a :class:`~cosq.decision.base.DecisionRule`: that contract takes
    binary ``Certainty`` labels, and pretending a graded signal fits it would hide the
    difference this class exists to expose. It returns the same
    :class:`~cosq.decision.base.DecisionOutcome`, now carrying ``score``.

    ``on_empty`` fixes what happens when stage 1 produced no parseable sub-facts. As
    with the binary rule the default is ``"abstain"``, and it is explicit because it
    moves AR.
    """

    name = "confidence"

    def __init__(
        self,
        threshold: float = 0.8,
        aggregator: str = "minimum",
        on_empty: str = "abstain",
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if aggregator not in AGGREGATORS:
            known = ", ".join(sorted(AGGREGATORS))
            raise ValueError(f"unknown aggregator {aggregator!r}; available: {known}")
        if on_empty not in ("abstain", "answer"):
            raise ValueError(f"on_empty must be 'abstain' or 'answer', got {on_empty!r}")
        self.threshold = threshold
        self.aggregator = aggregator
        self.on_empty = on_empty

    def decide(self, confidences: Sequence[float]) -> DecisionOutcome:
        n = len(confidences)
        if n == 0:
            return DecisionOutcome(
                decision="abstain" if self.on_empty == "abstain" else "answer",
                uncertain_fraction=1.0,
                n_needs=0,
                n_uncertain=0,
                reason=f"no parseable sub-facts; on_empty={self.on_empty}",
                score=None,
            )
        score = aggregate(confidences, self.aggregator)
        # Reported for continuity with the binary rule: sub-facts below the threshold
        # are the ones a "certain / uncertain" question would have flagged.
        n_uncertain = sum(1 for c in confidences if c < self.threshold)
        abstain = score < self.threshold
        return DecisionOutcome(
            decision="abstain" if abstain else "answer",
            uncertain_fraction=n_uncertain / n,
            n_needs=n,
            n_uncertain=n_uncertain,
            reason=(
                f"{self.aggregator}={score:.3f} "
                f"{'<' if abstain else '>='} threshold={self.threshold:.3f}"
            ),
            score=score,
        )

    def __repr__(self) -> str:
        return (
            f"ConfidenceRule(threshold={self.threshold!r}, "
            f"aggregator={self.aggregator!r}, on_empty={self.on_empty!r})"
        )
