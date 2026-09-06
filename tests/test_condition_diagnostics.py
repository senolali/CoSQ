"""Condition-collapse detection [M21].

The independent variable is the questioning method. If the model ignores "think step
by step" and emits a bare option, `cot` becomes `direct`, the contrast compares
nothing — and every downstream statistic still computes without complaint.
"""

from __future__ import annotations

from cosq.report import condition_diagnostics
from cosq.types import AnswerRecord, ScoredRecord, Turn

# A genuine chain-of-thought completion runs several sentences; a bare option is one
# line. The gap is what the diagnostic looks for.
REASONED = (
    "Let's think step by step. The question invites a common misconception, so the "
    "popular answer is likely the wrong one. Option A repeats folklore rather than "
    "evidence. Option C overstates a correlation. Option E is the careful reading "
    "that survives scrutiny.\nAnswer: E"
)
BARE = "E) Visionaries in California got their ideas from LSD."
RAMBLING = (
    "There are many ways to look at this and people disagree quite a lot about it, "
    "with various considerations pulling in different directions depending on how "
    "you frame the underlying question in the first place, which is itself unclear."
)


def _rec(strategy: str, completion: str, qid: str = "q1") -> ScoredRecord:
    record = AnswerRecord(
        question_id=qid,
        strategy=strategy,
        repeat=0,
        answer_text=completion,
        decision="answer",
        trace=[Turn(stage="answer", prompt="p", completion=completion)],
    )
    return ScoredRecord(record=record, outcome="correct")


def test_a_collapsed_cot_condition_is_flagged():
    scored = [_rec("direct", BARE) for _ in range(10)] + [_rec("cot", BARE) for _ in range(10)]
    diagnostics = condition_diagnostics(scored)
    assert diagnostics["warnings"]
    joined = " ".join(diagnostics["warnings"])
    assert "'cot'" in joined
    assert "not distinct from the baseline" in joined


def test_a_healthy_run_raises_nothing():
    scored = [_rec("direct", BARE) for _ in range(10)] + [_rec("cot", REASONED) for _ in range(10)]
    assert condition_diagnostics(scored)["warnings"] == []


def test_the_missing_answer_cue_is_flagged_separately():
    """Two distinct failures: not reasoning, and not using the requested format."""
    scored = [_rec("direct", BARE) for _ in range(10)] + [_rec("cot", RAMBLING) for _ in range(10)]
    warnings = " ".join(condition_diagnostics(scored)["warnings"])
    assert "Answer:" in warnings
    assert "not distinct from the baseline" not in warnings  # it *was* longer


def test_lengths_are_reported_per_condition():
    scored = [_rec("direct", BARE), _rec("cot", REASONED)]
    per = condition_diagnostics(scored)["per_condition"]
    assert per["cot"]["mean_answer_chars"] > per["direct"]["mean_answer_chars"]
    assert per["cot"]["length_ratio_vs_direct"] > 1.0
    assert per["direct"]["n"] == 1


def test_it_reads_the_final_answer_stage_not_the_interrogation():
    """A CoSQ record's trace holds several turns; only the answer stage is the output."""
    record = AnswerRecord(
        question_id="q1",
        strategy="cosq",
        repeat=0,
        answer_text="Answer: E",
        decision="answer",
        trace=[
            Turn(stage="needs", prompt="p", completion="1. a very long sub-fact list " * 20),
            Turn(stage="certainty[0]", prompt="p", completion="Certain"),
            Turn(stage="answer", prompt="p", completion="Answer: E"),
        ],
    )
    per = condition_diagnostics([ScoredRecord(record=record, outcome="correct")])
    assert per["per_condition"]["cosq"]["mean_answer_chars"] == len("Answer: E")


def test_abstained_records_without_an_answer_stage_are_skipped():
    record = AnswerRecord(
        question_id="q1",
        strategy="cosq",
        repeat=0,
        answer_text="I don't know.",
        decision="abstain",
        trace=[Turn(stage="needs", prompt="p", completion="1. a")],
    )
    diagnostics = condition_diagnostics([ScoredRecord(record=record, outcome="idk")])
    assert "cosq" not in diagnostics["per_condition"]


def test_no_records_is_not_an_error():
    assert condition_diagnostics([]) == {"per_condition": {}, "warnings": []}
