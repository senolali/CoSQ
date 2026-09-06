"""The decision-rule contract."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass

from cosq.types import Certainty, Decision


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    """A decision plus the quantities that produced it, kept for the audit trail."""

    decision: Decision
    uncertain_fraction: float
    n_needs: int
    n_uncertain: int
    reason: str
    #: Aggregate confidence, when the rule works on a graded signal. ``None`` for the
    #: binary rules, whose decision variable is the uncertain fraction above.
    score: float | None = None


class DecisionRule(abc.ABC):
    """Maps stage-2 certainty labels onto answer-or-abstain.

    Implementations must be pure functions: no I/O, no model calls, no randomness.
    That is what lets the whole τ sweep be recomputed offline from stored records.
    """

    name: str

    @abc.abstractmethod
    def decide(self, certainties: Sequence[Certainty]) -> DecisionOutcome:
        """Decide whether to answer, given one certainty label per sub-fact."""
