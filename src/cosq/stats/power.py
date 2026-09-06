"""Power analysis for the test that is actually run [M5].

The proposal's ``n >= 85`` corresponds to a paired t-test on continuous data. The
planned analysis is a paired test on *binary* outcomes, whose power depends on the
number of discordant pairs rather than on n. These helpers make that visible during
a run instead of after it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as _sp

#: Below this discordance rate a non-significant result is reported as underpowered,
#: never as a null. The two are different claims.
UNDERPOWERED_DISCORDANCE = 0.15


def mcnemar_power(n_discordant: int, pi: float, alpha: float = 0.05) -> float:
    """Power of the exact McNemar test.

    ``pi`` is the true probability that a discordant pair favours the treatment.
    """
    if n_discordant <= 0:
        return 0.0
    if not 0.0 <= pi <= 1.0:
        raise ValueError(f"pi must be in [0, 1], got {pi}")
    counts = np.arange(n_discordant + 1)
    p_values = np.array([_sp.binomtest(int(k), n_discordant, 0.5).pvalue for k in counts])
    return float(_sp.binom.pmf(counts, n_discordant, pi)[p_values <= alpha].sum())


def required_discordant_pairs(pi: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """Smallest number of discordant pairs reaching ``power`` at ``pi``."""
    for n in range(2, 2000):
        if mcnemar_power(n, pi, alpha) >= power:
            return n
    raise ValueError(f"no feasible n below 2000 for pi={pi}, power={power}")


@dataclass(frozen=True, slots=True)
class DiscordanceReport:
    n: int
    n_discordant: int
    discordance_rate: float
    achieved_power_pi80: float
    underpowered: bool

    def verdict(self) -> str:
        if self.underpowered:
            return (
                f"UNDERPOWERED: {self.n_discordant}/{self.n} discordant pairs "
                f"({self.discordance_rate:.1%}). A non-significant result here must be "
                "reported as underpowered, not as a null."
            )
        return (
            f"{self.n_discordant}/{self.n} discordant pairs ({self.discordance_rate:.1%}); "
            f"power {self.achieved_power_pi80:.2f} at pi=0.80."
        )


def discordance_report(a: Sequence[bool], b: Sequence[bool]) -> DiscordanceReport:
    """Primary diagnostic: how much information the paired comparison actually has."""
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    if x.shape != y.shape:
        raise ValueError(f"paired inputs must match: {x.shape} vs {y.shape}")
    n_discordant = int(np.sum(x != y))
    rate = n_discordant / x.size if x.size else 0.0
    return DiscordanceReport(
        n=int(x.size),
        n_discordant=n_discordant,
        discordance_rate=rate,
        achieved_power_pi80=mcnemar_power(n_discordant, 0.80),
        underpowered=rate < UNDERPOWERED_DISCORDANCE,
    )
