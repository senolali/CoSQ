"""Scoring and metrics. Pure functions over records — no model calls, ever."""

from cosq.eval.metrics import Counts, MetricSet, compute_metrics, count_outcomes
from cosq.eval.scorer import score_record, score_records
from cosq.eval.selective import (
    SelectivePoint,
    aurc,
    coverage_matched_risk,
    random_abstention_risk,
    risk_coverage_curve,
    selective_point,
)

__all__ = [
    "Counts",
    "MetricSet",
    "SelectivePoint",
    "aurc",
    "compute_metrics",
    "count_outcomes",
    "coverage_matched_risk",
    "random_abstention_risk",
    "risk_coverage_curve",
    "score_record",
    "score_records",
    "selective_point",
]
