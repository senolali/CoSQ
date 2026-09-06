"""Multiple-comparison control for the secondary family [M10]."""

from __future__ import annotations

from collections.abc import Sequence


def holm_bonferroni(
    p_values: Sequence[float], alpha: float = 0.05
) -> list[tuple[float, float, bool]]:
    """Holm's sequentially rejective procedure (Holm, 1979).

    Returns ``(p, adjusted_p, reject)`` in the caller's original order. Adjusted
    values are made monotone, so a later hypothesis is never reported as more
    significant than an earlier, smaller one.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])

    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (n - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return [(p_values[i], adjusted[i], adjusted[i] < alpha) for i in range(n)]
