"""Effect sizes and confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta: P(a > b) - P(a < b). Pre-registered effect size.

    On a binary outcome this collapses algebraically to the difference in
    proportions, so the pre-registered ``delta >= 0.3`` criterion amounts to
    demanding a 30 percentage-point absolute reduction in HR [M6]. Reported
    alongside :func:`risk_difference` and :func:`mcnemar_odds_ratio`, which are
    interpretable for binary data.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError("Cliff's delta needs non-empty inputs")
    comparisons = np.sign(x[:, None] - y[None, :])
    return float(comparisons.sum() / (x.size * y.size))


def risk_difference(
    a: Sequence[bool], b: Sequence[bool], *, n_boot: int = 10_000, seed: int = 1002
) -> tuple[float, tuple[float, float]]:
    """``rate(a) - rate(b)`` with a paired bootstrap 95% CI.

    Questions are resampled as *pairs*, preserving the within-subject design; an
    unpaired bootstrap would overstate the uncertainty.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired inputs must match: {x.shape} vs {y.shape}")
    point = float(x.mean() - y.mean())

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x.size, size=(n_boot, x.size))
    draws = x[indices].mean(axis=1) - y[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return point, (float(low), float(high))


def mcnemar_odds_ratio(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Odds ratio from the discordant pairs, ``b_count / c_count``.

    ``inf`` when every discordance favours ``a``; ``nan`` when there are none.
    """
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    b_count = int(np.sum(x & ~y))
    c_count = int(np.sum(~x & y))
    if b_count == 0 and c_count == 0:
        return float("nan")
    if c_count == 0:
        return float("inf")
    return b_count / c_count


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = lambda arr: float(arr.mean()),
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 1002,
) -> tuple[float, tuple[float, float]]:
    """Percentile bootstrap CI for any statistic. Seeded, so runs reproduce."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot bootstrap over zero values")
    point = statistic(array)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_boot, array.size))
    draws = np.array([statistic(array[row]) for row in indices])
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), (float(low), float(high))
