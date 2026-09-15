import pytest

from cosq.backends.mock import MockBackend
from cosq.eval.scorer import score_record
from cosq.strategies import CoTAbstainStrategy, CoTStrategy, DirectStrategy
from cosq.strategies.base import render_question
from cosq.types import GenerationParams, Question


def test_mc_answer_instructions_do_not_assume_four_options():
    question = Question(
        id="five-options",
        text="Which option is correct?",
        options=("first", "second", "third", "fourth", "fifth"),
        gold_index=4,
    )
    block = render_question(question)
    assert "E) fifth" in block


@pytest.mark.parametrize("strategy_cls", [DirectStrategy, CoTStrategy])
def test_forced_choice_mc_baselines_do_not_offer_abstention(strategy_cls):
    question = Question(
        id="forced-choice",
        text="Which option is correct?",
        options=("first", "second", "third"),
        gold_index=1,
    )
    strategy = strategy_cls(MockBackend(p_abstain=0.0), mc_output=True)
    prompt = strategy.answer(question, GenerationParams()).trace[0].prompt.upper()

    assert "ABSTAIN" not in prompt
    assert "I DON'T KNOW" not in prompt
    assert "I DO NOT KNOW" not in prompt


def test_cot_abstain_mc_baseline_explicitly_offers_abstention():
    question = Question(
        id="optional-abstention",
        text="Which option is correct?",
        options=("first", "second", "third"),
        gold_index=1,
    )
    strategy = CoTAbstainStrategy(MockBackend(p_abstain=0.0), mc_output=True)
    prompt = strategy.answer(question, GenerationParams()).trace[0].prompt.upper()

    assert "ABSTAIN" in prompt


@pytest.mark.parametrize("strategy_cls", [DirectStrategy, CoTStrategy])
@pytest.mark.parametrize("invalid_answer", ["ABSTAIN", "not an option"])
def test_forced_choice_mc_contract_violations_are_wrong(strategy_cls, invalid_answer):
    question = Question(
        id="forced-choice-scoring",
        text="Which option is correct?",
        options=("first", "second", "third"),
        gold_index=1,
    )
    record = strategy_cls(MockBackend(p_abstain=0.0), mc_output=True).answer(
        question, GenerationParams()
    )
    record.answer_text = invalid_answer

    assert score_record(record, question).outcome == "wrong"


def test_abstention_enabled_mc_output_remains_idk():
    question = Question(
        id="optional-abstention-scoring",
        text="Which option is correct?",
        options=("first", "second", "third"),
        gold_index=1,
    )
    record = CoTAbstainStrategy(MockBackend(p_abstain=0.0), mc_output=True).answer(
        question, GenerationParams()
    )
    record.answer_text = "ABSTAIN"

    assert score_record(record, question).outcome == "idk"
