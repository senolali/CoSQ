import math

import pytest

from conftest import scored
from cosq.eval import compute_metrics, count_outcomes


def test_counts_partition_the_sample():
    records = [
        scored("a", "correct"),
        scored("b", "wrong"),
        scored("c", "idk"),
        scored("d", "unparseable"),
    ]
    counts = count_outcomes(records)
    assert counts.total == 4
    metrics = compute_metrics(records)
    total_rate = (
        metrics.accuracy
        + metrics.hallucination_rate
        + metrics.abstention_rate
        + metrics.unparseable_rate
    )
    assert total_rate == pytest.approx(1.0)


def test_unparseable_is_not_counted_as_wrong():
    """A parser failure is a measurement failure, not a hallucination [M9]."""
    records = [scored("a", "correct"), scored("b", "unparseable")]
    metrics = compute_metrics(records)
    assert metrics.hallucination_rate == 0.0
    assert metrics.unparseable_rate == 0.5


def test_abstention_aware_accuracy_conditions_on_answering():
    records = [scored("a", "correct")] * 3 + [scored("b", "wrong")] + [scored("c", "idk")] * 6
    metrics = compute_metrics(records)
    assert metrics.accuracy == pytest.approx(0.3)
    assert metrics.abstention_aware_accuracy == pytest.approx(0.75)


def test_hr_is_gameable_by_abstention_but_phi_is_not():
    """The reason HR is never reported alone [M2]."""
    always_abstain = [scored(f"q{i}", "idk") for i in range(10)]
    metrics = compute_metrics(always_abstain)
    assert metrics.hallucination_rate == 0.0  # perfect by the letter of H1
    assert metrics.effective_reliability == 0.0  # and worth nothing


def test_effective_reliability_penalises_errors_as_much_as_it_rewards_answers():
    balanced = [scored("a", "correct"), scored("b", "wrong")]
    assert compute_metrics(balanced).effective_reliability == 0.0
    good = [scored("a", "correct"), scored("b", "correct"), scored("c", "wrong")]
    assert compute_metrics(good).effective_reliability == pytest.approx(1 / 3)


def test_abstention_aware_accuracy_is_nan_when_nothing_was_answered():
    metrics = compute_metrics([scored("a", "idk")])
    assert math.isnan(metrics.abstention_aware_accuracy)


def test_empty_input_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="zero records"):
        compute_metrics([])
