"""Turn raw answers into outcomes.

Runs entirely offline from ``records.jsonl``: no GPU, no network, no re-querying.
Fixing a parser and re-scoring every condition is therefore cheap, which is the
whole point of persisting raw output first.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from cosq.parsing import detect_abstention, match_open_answer, parse_mc_choice
from cosq.types import AnswerRecord, Outcome, Question, ScoredRecord


def score_record(record: AnswerRecord, question: Question) -> ScoredRecord:
    """Assign one of ``correct`` / ``wrong`` / ``idk`` / ``unparseable``.

    Precedence: an explicit decision to abstain, then an explicit refusal in the
    text, then option matching. ``unparseable`` — the parser could not map the answer
    to any option — is kept separate from ``wrong``, because a measurement failure is
    not a hallucination and folding the two together inflates HR [M9]. Expect
    materially more of it in open-ended mode, where matching is lexical rather than a
    label lookup [M22].
    """
    if record.decision == "abstain" or detect_abstention(record.answer_text):
        return ScoredRecord(record=record, outcome="idk", matched_index=None)

    # Which matcher applies is a property of how the question was presented, and the
    # record carries that, so a run stays re-scorable offline without its config.
    if record.meta.get("open_ended"):
        index = match_open_answer(record.answer_text, question.options, question=question.text)
    else:
        index = parse_mc_choice(record.answer_text, question.options)
    if index is None:
        if question.meta.get("scoring") == "short_answer":
            return ScoredRecord(record=record, outcome="wrong", matched_index=None)
        return ScoredRecord(record=record, outcome="unparseable", matched_index=None)

    if question.meta.get("all_options_correct"):
        outcome: Outcome = "correct"
    else:
        outcome = "correct" if index == question.gold_index else "wrong"
    return ScoredRecord(record=record, outcome=outcome, matched_index=index)


def score_records(
    records: Iterable[AnswerRecord], questions: Mapping[str, Question]
) -> list[ScoredRecord]:
    """Score many records. Raises on an unknown question id rather than skipping it."""
    scored = []
    for record in records:
        question = questions.get(record.question_id)
        if question is None:
            raise KeyError(f"no question with id {record.question_id!r} in the dataset")
        scored.append(score_record(record, question))
    return scored
