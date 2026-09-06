import pytest

from conftest import scored
from cosq.eval import (
    aurc,
    coverage_matched_risk,
    random_abstention_risk,
    selective_point,
)


def _baseline():
    # 10 questions: right on q0-q3, wrong on q4-q9.
    return [scored(f"q{i}", "correct" if i < 4 else "wrong") for i in range(10)]


def test_selective_point_measures_risk_among_answered():
    point = selective_point(_baseline())
    assert point.coverage == 1.0
    assert point.risk == pytest.approx(0.6)


def test_coverage_matching_exposes_indiscriminate_abstention():
    """A method that abstains on the questions it would have got *right* gains
    nothing under coverage matching — which is what H2 tests [M2]."""
    # Abstains on q0-q5, answers q6-q9 (all wrong).
    bad = [scored(f"q{i}", "idk" if i < 6 else "wrong") for i in range(10)]
    matched = coverage_matched_risk(_baseline(), bad)
    assert selective_point(bad).risk == 1.0
    assert matched.risk == 1.0  # baseline was equally wrong there: no discrimination


def test_coverage_matching_rewards_selective_abstention():
    # Abstains exactly on the six it would have got wrong.
    good = [scored(f"q{i}", "correct" if i < 4 else "idk") for i in range(10)]
    matched = coverage_matched_risk(_baseline(), good)
    assert selective_point(good).risk == 0.0
    assert matched.risk == 0.0
    # The baseline's *overall* risk was 0.6; matching removes the volume advantage.
    assert selective_point(_baseline()).risk == pytest.approx(0.6)


def test_random_abstention_is_the_null_model():
    risk = random_abstention_risk(_baseline(), coverage=0.5, n_boot=500)
    assert risk == pytest.approx(0.6, abs=0.1)


def test_aurc_needs_a_curve_not_a_point():
    single = [selective_point(_baseline(), "a")]
    with pytest.raises(ValueError, match="two distinct coverage"):
        aurc(single)


def test_aurc_orders_methods_by_risk_at_matched_coverage():
    low = [scored(f"q{i}", "correct" if i < 8 else "idk") for i in range(10)]
    high = [scored(f"q{i}", "wrong" if i < 8 else "idk") for i in range(10)]
    points_low = [selective_point(low, "low"), selective_point(_baseline(), "full")]
    points_high = [selective_point(high, "high"), selective_point(_baseline(), "full")]
    assert aurc(points_low) < aurc(points_high)


def test_selective_point_on_nothing_is_an_error():
    with pytest.raises(ValueError, match="zero records"):
        selective_point([])


def test_a_tiny_matched_subset_is_reported_as_indistinguishable():
    """The failure mode M18 exists to prevent: two risks differing by 0.01 on a dozen
    questions must not be reported as an advantage."""
    from cosq.report.analysis import _matched_pair_test

    # 12 answered questions; treatment and baseline differ on one of them.
    treatment = [scored(f"q{i}", "correct" if i else "wrong") for i in range(12)]
    treatment += [scored(f"q{i}", "idk") for i in range(12, 100)]
    baseline = [scored(f"q{i}", "correct") for i in range(12)]
    baseline += [scored(f"q{i}", "wrong") for i in range(12, 100)]

    out = _matched_pair_test(treatment, baseline)
    assert out["n_questions_matched"] == 12
    assert out["advantage_is_significant"] is False
    assert "INDISTINGUISHABLE" in out["verdict"]
