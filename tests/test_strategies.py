import pytest

from cosq.backends.mock import MockBackend
from cosq.decision import ThresholdRule
from cosq.registry import available, resolve
from cosq.strategies import CoSQStrategy, CoTAbstainStrategy, CoTStrategy, DirectStrategy


def test_every_condition_is_registered():
    assert set(available("strategy")) == {
        "direct",
        "cot",
        "cot_abstain",
        "cosq",
        "cosq_gate",  # stage-3 ablation [M19]
        "cosq_graded",  # graded confidence instead of a conjunction [M20]
        "cosq_graded_gate",
        "cosq_grounded",
        "cosq_grounded_adaptive",
        "cosq_critical_grounded",
    }


@pytest.mark.parametrize("cls", [DirectStrategy, CoTStrategy, CoTAbstainStrategy])
def test_baselines_issue_exactly_one_call(cls, question, params):
    backend = MockBackend()
    record = cls(backend).answer(question, params)
    assert backend.calls == 1
    assert len(record.trace) == 1
    assert record.trace[0].completion == record.answer_text


def test_only_cot_abstain_is_told_it_may_abstain(question, params):
    """The whole point of the fourth condition [M3]."""
    assert DirectStrategy(MockBackend()).answer(question, params).meta["allows_abstention"] is False
    assert CoTStrategy(MockBackend()).answer(question, params).meta["allows_abstention"] is False
    assert (
        CoTAbstainStrategy(MockBackend()).answer(question, params).meta["allows_abstention"] is True
    )


def test_baselines_without_permission_never_record_an_abstention(question, params):
    backend = MockBackend(responses={"Answer:": "I don't know."})
    record = CoTStrategy(backend).answer(question, params)
    assert record.decision == "answer"  # the prompt did not offer the option


def test_cot_abstain_records_an_explicit_refusal(question, params):
    backend = MockBackend(responses={"Answer:": "I don't know."})
    record = CoTAbstainStrategy(backend).answer(question, params)
    assert record.decision == "abstain"


def test_cosq_runs_all_three_stages(question, params):
    backend = MockBackend(
        responses={
            "Required information:": "1. the canonical count\n2. modern physiology",
            "Are you certain": "Certain",
            "Using only that information": "Answer: B",
        }
    )
    record = CoSQStrategy(backend, rule=ThresholdRule(tau=0.0)).answer(question, params)
    stages = [turn.stage for turn in record.trace]
    assert stages == ["needs", "certainty[0]", "certainty[1]", "answer"]
    assert record.needs == ["the canonical count", "modern physiology"]
    assert record.certainties == ["certain", "certain"]
    assert record.decision == "answer"


def test_cosq_abstains_without_calling_the_model_again(question, params):
    backend = MockBackend(
        responses={
            "Required information:": "1. a\n2. b",
            "Are you certain": "Uncertain",
        }
    )
    record = CoSQStrategy(backend, rule=ThresholdRule(tau=0.0)).answer(question, params)
    assert record.decision == "abstain"
    assert [turn.stage for turn in record.trace] == ["needs", "certainty[0]", "certainty[1]"]
    assert backend.calls == 3  # no stage-3 generation
    assert record.answer_text == "I don't know."


def test_unreadable_certainty_counts_as_uncertain_and_is_tallied(question, params):
    """A label we could not read is not evidence of knowledge; the rate is recorded."""
    backend = MockBackend(responses={"Required information:": "1. a", "Are you certain": "banana"})
    record = CoSQStrategy(backend).answer(question, params)
    assert record.certainties == ["uncertain"]
    assert record.meta["unparseable_certainties"] == 1
    assert record.decision == "abstain"


def test_tau_changes_the_decision_on_identical_model_output(question, params):
    """The policy lives in the control layer, not in the model."""
    # Keyed on the exact stage-2 prompt for "alpha", so exactly one of four
    # sub-facts comes back uncertain: u = 0.25.
    responses = {
        "Using only that information": "Answer: B",
        "information:\n\nalpha": "Uncertain",
        "Required information:": "1. alpha\n2. beta\n3. gamma\n4. delta",
        "Are you certain": "Certain",
    }
    strict = CoSQStrategy(MockBackend(responses=responses), rule=ThresholdRule(0.0))
    loose = CoSQStrategy(MockBackend(responses=responses), rule=ThresholdRule(0.5))
    strict_record = strict.answer(question, params)
    loose_record = loose.answer(question, params)
    assert strict_record.certainties == loose_record.certainties
    assert strict_record.certainties.count("uncertain") == 1
    assert strict_record.meta["uncertain_fraction"] == 0.25
    assert strict_record.decision == "abstain"
    assert loose_record.decision == "answer"


def test_cosq_records_the_sub_fact_count_for_the_confound_analysis(question, params):
    """AR must be reportable conditioned on n [M7]."""
    backend = MockBackend(responses={"Required information:": "1. a\n2. b\n3. c"})
    record = CoSQStrategy(backend).answer(question, params)
    assert record.n_needs == 3
    assert record.meta["n_needs"] == 3


def test_strategies_resolve_by_name(question, params):
    strategy = resolve("strategy", "cosq")(MockBackend())
    assert strategy.answer(question, params).strategy == "cosq"
