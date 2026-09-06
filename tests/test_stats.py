import math

import pytest
from scipy.stats import binomtest

from cosq.stats import (
    cliffs_delta,
    cochran_q,
    discordance_report,
    holm_bonferroni,
    mcnemar_exact,
    mcnemar_odds_ratio,
    mcnemar_power,
    required_discordant_pairs,
    risk_difference,
    shapiro_wilk,
    wilcoxon_signed_rank,
)


def test_mcnemar_matches_the_exact_binomial_on_discordant_pairs():
    a = [True] * 10 + [False] * 2 + [True] * 50
    b = [False] * 10 + [True] * 2 + [True] * 50
    result = mcnemar_exact(a, b)
    assert result.detail["b"] == 10
    assert result.detail["c"] == 2
    assert result.p_value == pytest.approx(binomtest(10, 12, 0.5).pvalue)


def test_mcnemar_ignores_concordant_pairs():
    """Power depends on discordant pairs, not n — adding agreement changes nothing [M5]."""
    a, b = [True] * 8 + [False] * 2, [False] * 8 + [True] * 2
    small = mcnemar_exact(a, b)
    large = mcnemar_exact(a + [True] * 500, b + [True] * 500)
    assert small.p_value == pytest.approx(large.p_value)
    assert large.n == 510
    assert large.detail["n_discordant"] == 10


def test_mcnemar_with_no_discordance_is_undefined_not_significant():
    result = mcnemar_exact([True] * 10, [True] * 10)
    assert result.p_value == 1.0
    assert "undefined" in result.detail["note"]


def test_cochran_q_matches_a_hand_computation():
    # cols = 8,5,0; rows = 2*5, 1*3, 0*2; Q = 2*(3*89 - 169)/(39 - 23) = 12.25
    result = cochran_q([True] * 8 + [False] * 2, [True] * 5 + [False] * 5, [False] * 10)
    assert result.statistic == pytest.approx(12.25)
    assert result.detail["df"] == 2


def test_cochran_q_needs_three_conditions():
    with pytest.raises(ValueError, match="at least 3"):
        cochran_q([True], [False])


def test_cliffs_delta_on_binary_data_equals_the_risk_difference():
    """Which is why delta >= 0.3 means a 30-point absolute HR reduction [M6]."""
    a = [1.0] * 20 + [0.0] * 80  # p = .20
    b = [1.0] * 35 + [0.0] * 65  # p = .35
    assert cliffs_delta(a, b) == pytest.approx(0.20 - 0.35, abs=1e-12)


def test_risk_difference_ci_brackets_the_point_estimate():
    a = [True] * 20 + [False] * 80
    b = [True] * 40 + [False] * 60
    point, (low, high) = risk_difference(a, b, n_boot=2000)
    assert point == pytest.approx(-0.20)
    assert low < point < high


def test_risk_difference_is_seeded_and_reproducible():
    a, b = [True] * 20 + [False] * 80, [True] * 40 + [False] * 60
    assert risk_difference(a, b, n_boot=500) == risk_difference(a, b, n_boot=500)


def test_odds_ratio_edge_cases():
    assert math.isnan(mcnemar_odds_ratio([True] * 5, [True] * 5))
    assert mcnemar_odds_ratio([True] * 3 + [False], [False] * 3 + [False]) == float("inf")


def test_wilcoxon_is_still_reported_but_reports_its_ties():
    """Pre-registered, so never dropped — but its weakness is made visible [M4]."""
    a = [0.0, 1 / 3, 2 / 3, 1.0] * 5
    b = [1 / 3, 1 / 3, 2 / 3, 2 / 3] * 5
    result = wilcoxon_signed_rank(a, b)
    assert result.detail["n_nonzero"] == 10
    assert 0.0 <= result.p_value <= 1.0


def test_wilcoxon_on_identical_inputs_is_undefined_not_significant():
    result = wilcoxon_signed_rank([0.5] * 10, [0.5] * 10)
    assert result.p_value == 1.0
    assert result.detail["n_nonzero"] == 0


def test_shapiro_needs_three_points():
    with pytest.raises(ValueError, match="at least 3"):
        shapiro_wilk([1.0, 2.0])


def test_holm_is_monotone_and_less_strict_than_bonferroni():
    result = holm_bonferroni([0.01, 0.04, 0.03])
    adjusted = [adj for _, adj, _ in result]
    assert adjusted[0] == pytest.approx(0.03)  # 3 * 0.01
    assert adjusted[1] == adjusted[2] == pytest.approx(0.06)  # made monotone
    assert [reject for _, _, reject in result] == [True, False, False]


def test_holm_handles_the_empty_family():
    assert holm_bonferroni([]) == []


def test_power_table_matches_the_published_figures():
    """The numbers quoted in docs/methodology-notes.md and README §7.7 [M5]."""
    assert mcnemar_power(20, 0.80) == pytest.approx(0.80, abs=0.005)
    assert required_discordant_pairs(0.80) == 20
    assert required_discordant_pairs(0.75) == 30
    assert mcnemar_power(10, 0.70) == pytest.approx(0.15, abs=0.01)


def test_low_discordance_is_flagged_as_underpowered_not_null():
    report = discordance_report([True] * 98 + [False] * 2, [True] * 100)
    assert report.underpowered is True
    assert "UNDERPOWERED" in report.verdict()


def test_healthy_discordance_is_not_flagged():
    report = discordance_report([True] * 70 + [False] * 30, [True] * 100)
    assert report.underpowered is False
