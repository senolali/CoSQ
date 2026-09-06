"""The fraction-of-uncertain-sub-facts rule."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from cosq.decision.base import DecisionOutcome, DecisionRule
from cosq.registry import register
from cosq.types import Certainty

EmptyPolicy = Literal["abstain", "answer"]


@register("decision", "threshold")
class ThresholdRule(DecisionRule):
    """Abstain when the fraction of uncertain sub-facts exceeds ``tau``.

    ``tau = 0`` is the strict rule and the primary configuration: a single uncertain
    sub-fact triggers abstention. Larger values relax it. The rule is expressed as a
    *fraction* rather than a count so it normalises over how many sub-facts stage 1
    happened to enumerate — under a strict count-based rule the abstention
    probability would grow as ``1 - (1-p)**n`` with list length alone [M7].

    ``on_empty`` fixes what happens when stage 1 yielded no parseable sub-facts.
    The default is ``"abstain"``: with nothing established, there is no grounding to
    answer from. It is explicit rather than implicit because it moves AR, and any
    choice that moves a reported metric has to be pre-specified.
    """

    name = "threshold"

    def __init__(self, tau: float = 0.0, on_empty: EmptyPolicy = "abstain") -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        self.tau = tau
        self.on_empty = on_empty

    def decide(self, certainties: Sequence[Certainty]) -> DecisionOutcome:
        n = len(certainties)
        if n == 0:
            return DecisionOutcome(
                decision="abstain" if self.on_empty == "abstain" else "answer",
                uncertain_fraction=1.0,
                n_needs=0,
                n_uncertain=0,
                reason=f"no parseable sub-facts; on_empty={self.on_empty}",
            )
        n_uncertain = sum(1 for c in certainties if c == "uncertain")
        fraction = n_uncertain / n
        abstain = fraction > self.tau
        return DecisionOutcome(
            decision="abstain" if abstain else "answer",
            uncertain_fraction=fraction,
            n_needs=n,
            n_uncertain=n_uncertain,
            reason=f"u={fraction:.3f} {'>' if abstain else '<='} tau={self.tau:.3f}",
        )

    def __repr__(self) -> str:
        return f"ThresholdRule(tau={self.tau!r}, on_empty={self.on_empty!r})"
