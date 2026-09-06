"""Shared fixtures. Every test runs against the mock backend: no network, no GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from cosq.backends.mock import MockBackend
from cosq.data.jsonl import JsonlDataset
from cosq.types import AnswerRecord, GenerationParams, Question, ScoredRecord

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def params() -> GenerationParams:
    return GenerationParams()


@pytest.fixture
def question() -> Question:
    return Question(
        id="q1",
        text="How many senses do humans have?",
        options=("five", "more than five", "three"),
        gold_index=1,
    )


@pytest.fixture
def questions() -> list[Question]:
    return JsonlDataset(FIXTURES / "mini.jsonl").load()


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend()


def scored(qid: str, outcome: str, strategy: str = "s", repeat: int = 0) -> ScoredRecord:
    """Build a ScoredRecord directly, for metric and statistics tests."""
    return ScoredRecord(
        record=AnswerRecord(
            question_id=qid,
            strategy=strategy,
            repeat=repeat,
            answer_text="",
            decision="abstain" if outcome == "idk" else "answer",
        ),
        outcome=outcome,  # type: ignore[arg-type]
    )
