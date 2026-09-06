"""Markdown table rendering for generated reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    body = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    body.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(body)


def _fmt(value: float, digits: int = 3) -> str:
    return "---" if value != value else f"{value:.{digits}f}"  # NaN check


def metrics_table(metrics: Mapping[str, Mapping[str, Any]]) -> str:
    """The per-condition results table, in the schema the README commits to."""
    headers = [
        "Condition",
        "n",
        "HR down",
        "AR",
        "AA up",
        "Coverage",
        "Accuracy overall up",
        "Phi up",
        "Unparseable",
    ]
    rows = []
    for name, m in metrics.items():
        rows.append(
            [
                name,
                m["n"],
                _fmt(m["hallucination_rate"]),
                _fmt(m["abstention_rate"]),
                _fmt(m["abstention_aware_accuracy"]),
                _fmt(m["coverage"]),
                _fmt(m["accuracy"]),
                _fmt(m["effective_reliability"]),
                _fmt(m["unparseable_rate"]),
            ]
        )
    return render_markdown_table(headers, rows)


def metrics_note() -> str:
    return (
        "For abstaining systems, report answered accuracy as "
        "`AA = correct / (correct + wrong)`, read together with `Coverage`. "
        "`Accuracy overall = correct / N` is coverage-inclusive and therefore "
        "penalizes deliberate abstention. `Unparseable` marks measurement failure, "
        "not an explicit abstention."
    )
