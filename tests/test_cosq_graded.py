"""Graded-confidence CoSQ: a continuous decision variable instead of a conjunction."""

from __future__ import annotations

import pytest

from cosq.backends.mock import MockBackend
from cosq.decision import AGGREGATORS, ConfidenceRule, ThresholdRule, aggregate
from cosq.parsing.stages import parse_confidence
from cosq.registry import available
from cosq.strategies import CoSQGradedGateStrategy, CoSQGradedStrategy

# --- the parser ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("85", 0.85),
        ("85%", 0.85),
        ("0.85", 0.85),
        (".75", 0.75),
        ("Confidence: 90", 0.90),
        ("I would say about 70%", 0.70),
        ("100", 1.0),
        ("0", 0.0),
    ],
)
def test_parse_confidence_accepts_the_common_conventions(text, expected):
    assert parse_confidence(text) == pytest.approx(expected)


def test_a_bare_one_is_one_percent_not_full_confidence():
    """The prompt asks for 0-100. Reading "1" as 100% would inflate coverage in the
    model's favour on exactly the replies that are most ambiguous."""
    assert parse_confidence("1") == pytest.approx(0.01)
    assert parse_confidence("100") == pytest.approx(1.0)


def test_it_falls_back_to_the_binary_wording():
    assert parse_confidence("Certain") == 1.0
    assert parse_confidence("Uncertain") == 0.0


def test_unreadable_replies_return_none_rather_than_a_default():
    assert parse_confidence("banana") is None
    assert parse_confidence("") is None


def test_values_are_clamped_to_the_unit_interval():
    assert parse_confidence("150") == 1.0


# --- aggregators -----------------------------------------------------------


def test_every_aggregator_is_documented_and_agrees_at_the_extremes():
    """All four must map full confidence to 1 and any zero to 0 — otherwise they are
    not aggregating the same quantity."""
    for name in AGGREGATORS:
        assert aggregate([1.0, 1.0, 1.0], name) == pytest.approx(1.0)
        assert aggregate([1.0, 0.0], name) == pytest.approx(0.0) or name == "mean"


def test_aggregators_order_as_their_definitions_require():
    c = [0.9, 0.6, 0.8]
    assert aggregate(c, "minimum") == pytest.approx(0.6)
    assert aggregate(c, "mean") == pytest.approx(0.7666666, abs=1e-6)
    assert aggregate(c, "product") == pytest.approx(0.432)
    assert aggregate(c, "geometric_mean") == pytest.approx(0.432 ** (1 / 3))
    assert aggregate(c, "product") < aggregate(c, "geometric_mean") < aggregate(c, "mean")


def test_product_is_length_sensitive_and_geometric_mean_is_not():
    """Needing more facts really is riskier — whether that is signal or an artefact of
    how verbose the model was is the empirical question [M7]."""
    short, long = [0.9, 0.9], [0.9] * 6
    assert aggregate(long, "product") < aggregate(short, "product")
    assert aggregate(long, "geometric_mean") == pytest.approx(aggregate(short, "geometric_mean"))


def test_invalid_aggregator_and_inputs_are_rejected():
    with pytest.raises(ValueError, match="unknown aggregator"):
        aggregate([0.5], "median")
    with pytest.raises(ValueError, match="empty"):
        aggregate([], "mean")
    with pytest.raises(ValueError, match="outside"):
        aggregate([1.5], "mean")


# --- the equivalences that keep the pre-registered rule a special case ------


def test_minimum_at_threshold_one_reproduces_the_strict_tau_zero_rule():
    """Nothing pre-registered is discarded: τ=0 is this rule's corner case."""
    graded, binary = ConfidenceRule(1.0, "minimum"), ThresholdRule(0.0)
    for labels in (
        ["certain", "certain", "certain"],
        ["certain", "uncertain", "certain"],
        ["uncertain"],
    ):
        confidences = [1.0 if x == "certain" else 0.0 for x in labels]
        assert graded.decide(confidences).decision == binary.decide(labels).decision


@pytest.mark.parametrize("tau", [0.0, 0.15, 0.25, 0.34, 0.5])
@pytest.mark.parametrize(
    "labels",
    [
        ["certain"] * 4,
        ["certain", "certain", "certain", "uncertain"],
        ["certain", "uncertain", "certain", "uncertain"],
        ["uncertain"] * 4,
    ],
)
def test_mean_at_one_minus_tau_reproduces_the_threshold_rule(tau, labels):
    graded = ConfidenceRule(1 - tau, "mean")
    confidences = [1.0 if x == "certain" else 0.0 for x in labels]
    assert graded.decide(confidences).decision == ThresholdRule(tau).decide(labels).decision


def test_the_graded_rule_resolves_cases_the_binary_one_cannot_distinguish():
    """The point of the variant: two answer sets the conjunction calls identical."""
    binary = ThresholdRule(0.0)
    labels = ["certain", "certain", "uncertain"]
    assert binary.decide(labels).decision == "abstain"  # both of the below collapse here

    rule = ConfidenceRule(0.5, "minimum")
    assert rule.decide([0.95, 0.95, 0.60]).decision == "answer"  # barely unsure
    assert rule.decide([0.95, 0.95, 0.05]).decision == "abstain"  # genuinely lost


def test_empty_and_invalid_configurations():
    assert ConfidenceRule(on_empty="abstain").decide([]).decision == "abstain"
    assert ConfidenceRule(on_empty="answer").decide([]).decision == "answer"
    assert ConfidenceRule().decide([]).score is None
    for kwargs in ({"threshold": 1.5}, {"aggregator": "median"}, {"on_empty": "maybe"}):
        with pytest.raises(ValueError):
            ConfidenceRule(**kwargs)  # type: ignore[arg-type]


# --- the strategy ----------------------------------------------------------

RESPONSES = {
    "Required information:": "1. alpha\n2. beta\n3. gamma",
    "How confident are you": "85",
    "Using only that information": "Reasoning: ok.\nAnswer: B",
}


def test_registered():
    assert "cosq_graded" in available("strategy")
    assert "cosq_graded_gate" in available("strategy")
    assert "confidence" in available("decision")


def test_it_asks_for_a_number_not_a_label(question, params):
    record = CoSQGradedStrategy(MockBackend(responses=RESPONSES)).answer(question, params)
    stage2 = next(t for t in record.trace if t.stage == "confidence[0]")
    flattened = " ".join(stage2.prompt.split())
    assert "single integer from 0 to 100" in flattened
    assert "Certain" not in flattened  # the binary wording must be gone


def test_it_records_the_graded_signal(question, params):
    record = CoSQGradedStrategy(MockBackend(responses=RESPONSES)).answer(question, params)
    assert record.meta["confidences"] == [0.85, 0.85, 0.85]
    assert record.meta["aggregate_confidence"] == pytest.approx(0.85)
    assert record.certainties == ["certain"] * 3  # kept for comparability with cosq
    assert [t.stage for t in record.trace] == [
        "needs",
        "confidence[0]",
        "confidence[1]",
        "confidence[2]",
        "answer",
    ]


def test_the_threshold_decides_on_identical_model_output(question, params):
    low = CoSQGradedStrategy(
        MockBackend(responses=RESPONSES), rule=ConfidenceRule(0.5, "minimum")
    ).answer(question, params)
    high = CoSQGradedStrategy(
        MockBackend(responses=RESPONSES), rule=ConfidenceRule(0.9, "minimum")
    ).answer(question, params)
    assert low.meta["confidences"] == high.meta["confidences"]
    assert (low.decision, high.decision) == ("answer", "abstain")


def test_abstention_costs_no_extra_call(question, params):
    backend = MockBackend(
        responses={"Required information:": "1. alpha\n2. beta", "How confident are you": "10"}
    )
    record = CoSQGradedStrategy(backend, rule=ConfidenceRule(0.8, "minimum")).answer(
        question, params
    )
    assert record.decision == "abstain"
    assert backend.calls == 3
    assert record.answer_text == "I don't know."


def test_an_unreadable_confidence_counts_as_no_evidence_and_is_tallied(question, params):
    backend = MockBackend(
        responses={"Required information:": "1. alpha", "How confident are you": "banana"}
    )
    record = CoSQGradedStrategy(backend).answer(question, params)
    assert record.meta["confidences"] == [0.0]
    assert record.meta["unparseable_confidences"] == 1
    assert record.decision == "abstain"


def test_only_confident_sub_facts_reach_stage_three(question, params):
    backend = MockBackend(
        responses={
            "Using only that information": "Answer: B",
            "information:\n\nalpha": "20",
            "Required information:": "1. alpha\n2. beta\n3. gamma",
            "How confident are you": "95",
        }
    )
    record = CoSQGradedStrategy(backend, rule=ConfidenceRule(0.5, "mean")).answer(question, params)
    assert record.decision == "answer"
    answer_prompt = next(t for t in record.trace if t.stage == "answer").prompt
    assert "- beta" in answer_prompt and "- gamma" in answer_prompt
    assert "- alpha" not in answer_prompt


def test_builds_from_a_config():
    from cosq.config import StrategyConfig

    strategy = StrategyConfig(name="cosq_graded", threshold=0.6, aggregator="geometric_mean").build(
        MockBackend()
    )
    assert isinstance(strategy, CoSQGradedStrategy)
    assert strategy.confidence_rule.threshold == 0.6
    assert strategy.confidence_rule.aggregator == "geometric_mean"


def test_graded_gate_builds_from_a_config():
    from cosq.config import StrategyConfig

    strategy = StrategyConfig(name="cosq_graded_gate", threshold=0.6, aggregator="mean").build(
        MockBackend()
    )
    assert isinstance(strategy, CoSQGradedGateStrategy)
    assert strategy.confidence_rule.threshold == 0.6
    assert strategy.confidence_rule.aggregator == "mean"


def test_graded_gate_answers_with_the_plain_cot_prompt(question, params):
    from cosq.strategies.baselines import CoTStrategy

    responses = {
        "Required information:": "1. alpha\n2. beta",
        "How confident are you": "95",
        "Let's think step by step": "Reasoning: weighing it up.\nAnswer: B",
        "Using only that information": "Reasoning: from sub-facts.\nAnswer: A",
    }
    gate = CoSQGradedGateStrategy(
        MockBackend(responses=responses), rule=ConfidenceRule(0.5, "mean")
    ).answer(question, params)
    cot = CoTStrategy(MockBackend(responses=responses)).answer(question, params)
    gate_prompt = next(t for t in gate.trace if t.stage == "answer").prompt
    cot_prompt = next(t for t in cot.trace if t.stage == "answer").prompt
    assert gate.meta["aggregate_confidence"] == pytest.approx(0.95)
    assert gate_prompt == cot_prompt
    assert "You have confirmed that you are certain" not in gate_prompt


def test_invalid_unparseable_default_rejected():
    with pytest.raises(ValueError, match="unparseable_confidence"):
        CoSQGradedStrategy(MockBackend(), unparseable_confidence=2.0)
