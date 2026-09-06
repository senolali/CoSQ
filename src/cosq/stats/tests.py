"""Hypothesis tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as _sp


@dataclass(frozen=True, slots=True)
class TestResult:
    """A test's output, carrying enough context to report it honestly."""

    name: str
    statistic: float
    p_value: float
    n: int
    detail: dict[str, Any] = field(default_factory=dict)

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def shapiro_wilk(values: Sequence[float]) -> TestResult:
    """Normality check on paired differences, as pre-registered."""
    array = np.asarray(values, dtype=float)
    if array.size < 3:
        raise ValueError("Shapiro-Wilk needs at least 3 observations")
    statistic, p_value = _sp.shapiro(array)
    return TestResult("shapiro_wilk", float(statistic), float(p_value), int(array.size))


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Pre-registered paired comparison.

    Reported for every primary contrast because the funded proposal committed to it.
    Note its limitation on this data: the per-question outcome averaged over 3 repeats
    takes only the values {0, 1/3, 2/3, 1}, so ties dominate and zero differences are
    discarded, shrinking the effective sample. McNemar is the co-primary test [M4].
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired inputs must match: {x.shape} vs {y.shape}")
    nonzero = int(np.count_nonzero(x - y))
    if nonzero == 0:
        return TestResult(
            "wilcoxon",
            float("nan"),
            1.0,
            int(x.size),
            {"note": "all differences are zero; the test is undefined", "n_nonzero": 0},
        )
    statistic, p_value = _sp.wilcoxon(x, y)
    return TestResult(
        "wilcoxon", float(statistic), float(p_value), int(x.size), {"n_nonzero": nonzero}
    )


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> TestResult:
    """Co-primary test: exact McNemar on paired binary outcomes.

    ``a`` and ``b`` are per-question success indicators for two conditions. The exact
    test is a two-sided binomial test on the discordant pairs against p = 0.5, so its
    power depends on the *number of discordant pairs*, not on n — concordant questions
    carry no information [M5]. The counts are returned so that can be reported.
    """
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    if x.shape != y.shape:
        raise ValueError(f"paired inputs must match: {x.shape} vs {y.shape}")

    b_count = int(np.sum(x & ~y))  # a succeeds, b fails
    c_count = int(np.sum(~x & y))  # b succeeds, a fails
    n_discordant = b_count + c_count
    detail = {
        "b": b_count,
        "c": c_count,
        "n_discordant": n_discordant,
        "discordance_rate": n_discordant / x.size if x.size else 0.0,
    }
    if n_discordant == 0:
        return TestResult(
            "mcnemar_exact",
            float("nan"),
            1.0,
            int(x.size),
            {**detail, "note": "no discordant pairs; the test is undefined"},
        )
    result = _sp.binomtest(b_count, n_discordant, 0.5)
    return TestResult("mcnemar_exact", float(b_count), float(result.pvalue), int(x.size), detail)


def cochran_q(*conditions: Sequence[bool]) -> TestResult:
    """Omnibus test across k paired binary conditions.

    Run before pairwise McNemar so the four-condition comparison is handled coherently
    rather than as independent two-sample tests [M4].
    """
    if len(conditions) < 3:
        raise ValueError("Cochran's Q needs at least 3 conditions")
    matrix = np.asarray(conditions, dtype=float).T  # blocks x treatments
    if matrix.ndim != 2:
        raise ValueError("conditions must all be the same length")

    k = matrix.shape[1]
    col = matrix.sum(axis=0)
    row = matrix.sum(axis=1)
    total = float(matrix.sum())
    denominator = k * total - float(np.sum(row**2))
    if denominator == 0:
        return TestResult(
            "cochran_q",
            float("nan"),
            1.0,
            int(matrix.shape[0]),
            {"note": "every block is constant; the test is undefined", "k": k},
        )
    q = (k - 1) * (k * float(np.sum(col**2)) - total**2) / denominator
    p_value = float(_sp.chi2.sf(q, k - 1))
    return TestResult("cochran_q", float(q), p_value, int(matrix.shape[0]), {"k": k, "df": k - 1})
