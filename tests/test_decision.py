import pytest

from cosq.decision import ThresholdRule


def test_strict_rule_abstains_on_any_uncertainty():
    rule = ThresholdRule(tau=0.0)
    assert rule.decide(["certain", "certain", "certain"]).decision == "answer"
    assert rule.decide(["certain", "uncertain", "certain"]).decision == "abstain"


def test_threshold_relaxes_proportionally():
    certainties = ["certain", "certain", "uncertain", "certain"]  # u = 0.25
    assert ThresholdRule(tau=0.0).decide(certainties).decision == "abstain"
    assert ThresholdRule(tau=0.25).decide(certainties).decision == "answer"
    assert ThresholdRule(tau=0.2).decide(certainties).decision == "abstain"


def test_rule_normalises_over_list_length():
    """A fraction, not a count: one uncertain item out of ten is not the same as
    one out of two. Without this, list verbosity alone would drive abstention [M7]."""
    rule = ThresholdRule(tau=0.3)
    short = ["certain", "uncertain"]  # u = 0.5 -> abstain
    long = ["certain"] * 9 + ["uncertain"]  # u = 0.1 -> answer
    assert rule.decide(short).decision == "abstain"
    assert rule.decide(long).decision == "answer"


def test_empty_needs_policy_is_explicit():
    assert ThresholdRule(on_empty="abstain").decide([]).decision == "abstain"
    assert ThresholdRule(on_empty="answer").decide([]).decision == "answer"


def test_outcome_carries_its_reasoning():
    outcome = ThresholdRule(tau=0.0).decide(["certain", "uncertain"])
    assert outcome.n_needs == 2
    assert outcome.n_uncertain == 1
    assert outcome.uncertain_fraction == 0.5
    assert "tau" in outcome.reason


@pytest.mark.parametrize("tau", [-0.1, 1.5])
def test_invalid_tau_rejected(tau):
    with pytest.raises(ValueError, match="tau"):
        ThresholdRule(tau=tau)
