"""CoSQ-Gate: the self-questioning gate with stage 3 ablated away."""

from __future__ import annotations

import pytest

from cosq.backends.mock import MockBackend
from cosq.decision import ThresholdRule
from cosq.registry import available, resolve
from cosq.strategies import CoSQGateStrategy, CoSQStrategy

RESPONSES = {
    "Required information:": "1. alpha\n2. beta\n3. gamma",
    "Are you certain": "Certain",
    "Let's think step by step": "Reasoning: weighing it up.\nAnswer: B",
    "Using only that information": "Reasoning: from the sub-facts.\nAnswer: A",
}


def _gate(**kwargs):
    return CoSQGateStrategy(MockBackend(responses=RESPONSES), **kwargs)


def _cosq(**kwargs):
    return CoSQStrategy(MockBackend(responses=RESPONSES), **kwargs)


def test_registered_alongside_the_other_conditions():
    assert "cosq_gate" in available("strategy")
    assert resolve("strategy", "cosq_gate") is CoSQGateStrategy


def test_the_gate_is_identical_to_cosq(question, params):
    """Stages 1-2 and the decision must not differ — only what follows the decision."""
    gate = _gate().answer(question, params)
    cosq = _cosq().answer(question, params)
    assert gate.needs == cosq.needs
    assert gate.certainties == cosq.certainties
    assert gate.decision == cosq.decision
    assert gate.meta["uncertain_fraction"] == cosq.meta["uncertain_fraction"]
    assert [t.stage for t in gate.trace] == [t.stage for t in cosq.trace]


def test_it_answers_with_the_plain_cot_prompt(question, params):
    """The whole ablation: no sub-facts are carried into generation."""
    record = _gate().answer(question, params)
    answer_turn = next(t for t in record.trace if t.stage == "answer")
    assert "Let's think step by step" in answer_turn.prompt
    assert "You have confirmed that you are certain" not in answer_turn.prompt
    for need in record.needs:
        assert f"- {need}" not in answer_turn.prompt


def test_its_answer_prompt_matches_the_cot_condition_byte_for_byte(question, params):
    """Equality with `cot` is what makes the contrast interpretable: any accuracy
    difference on answered questions must come from the gate, not the wording."""
    from cosq.strategies.baselines import CoTStrategy

    gate = _gate().answer(question, params)
    cot = CoTStrategy(MockBackend(responses=RESPONSES)).answer(question, params)
    gate_prompt = next(t for t in gate.trace if t.stage == "answer").prompt
    cot_prompt = next(t for t in cot.trace if t.stage == "answer").prompt
    assert gate_prompt == cot_prompt


def test_cosq_still_passes_the_sub_facts(question, params):
    """The hook must not have changed CoSQ itself."""
    record = _cosq().answer(question, params)
    answer_turn = next(t for t in record.trace if t.stage == "answer")
    assert "You have confirmed that you are certain" in answer_turn.prompt
    assert "- alpha" in answer_turn.prompt


def test_abstention_path_is_unchanged_and_costs_no_extra_call(question, params):
    backend = MockBackend(
        responses={"Required information:": "1. alpha\n2. beta", "Are you certain": "Uncertain"}
    )
    record = CoSQGateStrategy(backend, rule=ThresholdRule(0.0)).answer(question, params)
    assert record.decision == "abstain"
    assert record.answer_text == "I don't know."
    assert [t.stage for t in record.trace] == ["needs", "certainty[0]", "certainty[1]"]
    assert backend.calls == 3


@pytest.mark.parametrize(("tau", "expected"), [(0.0, "abstain"), (0.5, "answer")])
def test_the_decision_rule_still_governs(question, params, tau, expected):
    responses = {
        "Using only that information": "Answer: A",
        "Let's think step by step": "Answer: B",
        "information:\n\nalpha": "Uncertain",
        "Required information:": "1. alpha\n2. beta\n3. gamma\n4. delta",
        "Are you certain": "Certain",
    }
    record = CoSQGateStrategy(MockBackend(responses=responses), rule=ThresholdRule(tau)).answer(
        question, params
    )
    assert record.meta["uncertain_fraction"] == 0.25
    assert record.decision == expected


def test_builds_from_a_config_with_tau():
    from cosq.config import StrategyConfig

    strategy = StrategyConfig(name="cosq_gate", tau=0.3).build(MockBackend())
    assert isinstance(strategy, CoSQGateStrategy)
    assert strategy.rule.tau == 0.3  # type: ignore[attr-defined]


def test_the_ablation_contrast_is_reported(tmp_path):
    """cosq vs cosq_gate must appear in the analysis, since it is the only contrast
    that isolates stage 3 [M19]."""
    from cosq.config import ExperimentConfig
    from cosq.eval import score_records
    from cosq.report import analyze_run
    from cosq.runner import run_experiment

    config = ExperimentConfig.from_mapping(
        {
            "name": "ablation",
            "model": {"id": "mock", "revision": "n/a", "backend": "mock"},
            "data": {"name": "jsonl", "path": "tests/fixtures/mini.jsonl", "n": 5, "seed": 1},
            "strategies": [
                "cot",
                {"name": "cosq", "tau": 0.5},
                {"name": "cosq_gate", "tau": 0.5},
            ],
            "repeats": 2,
        }
    )
    result = run_experiment(
        config,
        results_root=tmp_path / "runs",
        cache_path=None,
        backend=MockBackend(),
        allow_dirty=True,
    )
    scored = score_records(result.records, {q.id: q for q in result.questions})
    analysis = analyze_run(scored)

    ablation = next(c for c in analysis["secondary"] if c["baseline"] == "cosq_gate")
    assert ablation["treatment"] == "cosq"
    assert "stage 3" in ablation["measures"]


def test_the_gate_abstains_on_exactly_the_same_questions_as_cosq(question, params):
    """What makes the ablation clean: identical coverage, so the contrast cannot be
    confounded by which questions each condition chose to answer."""
    from cosq.data.jsonl import JsonlDataset

    questions = JsonlDataset("tests/fixtures/mini.jsonl").load()
    for q in questions:
        gate = CoSQGateStrategy(MockBackend(), rule=ThresholdRule(0.4)).answer(q, params)
        cosq = CoSQStrategy(MockBackend(), rule=ThresholdRule(0.4)).answer(q, params)
        assert gate.decision == cosq.decision
        assert gate.certainties == cosq.certainties
