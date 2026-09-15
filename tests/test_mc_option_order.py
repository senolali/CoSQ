from __future__ import annotations

from collections import Counter

import pytest

from cosq.data.base import (
    balance_option_positions,
    prepare_questions,
    validate_mc_option_positions,
)
from cosq.types import Question


def _questions(count: int = 12, option_count: int = 4) -> list[Question]:
    return [
        Question(
            id=f"q-{index:03d}",
            text=f"Question {index}",
            options=tuple(f"q{index}-option{option}" for option in range(option_count)),
            gold_index=0,
        )
        for index in range(count)
    ]


def test_balanced_order_preserves_gold_content_and_balances_positions() -> None:
    source = _questions()
    shuffled = balance_option_positions(source, seed=1002)

    assert Counter(question.gold_index for question in shuffled) == Counter(
        {0: 3, 1: 3, 2: 3, 3: 3}
    )
    for original, transformed in zip(source, shuffled, strict=True):
        assert transformed.gold == original.gold
        assert sorted(transformed.options) == sorted(original.options)
        assert transformed.meta["option_order"] == "balanced"
        assert transformed.options[transformed.gold_index] == original.gold


def test_balance_is_within_one_position_per_option_count_group() -> None:
    source = [
        *_questions(count=13, option_count=3),
        *[
            Question(
                id=f"four-{index:03d}",
                text=f"Four-option question {index}",
                options=tuple(f"four{index}-option{option}" for option in range(4)),
                gold_index=2,
            )
            for index in range(18)
        ],
    ]

    shuffled = balance_option_positions(source, seed=1002)
    for option_count in (3, 4):
        counts = Counter(
            question.gold_index
            for question in shuffled
            if len(question.options) == option_count
        )
        frequencies = [counts[position] for position in range(option_count)]
        assert max(frequencies) - min(frequencies) <= 1


def test_balanced_order_is_reproducible_and_seeded() -> None:
    source = _questions()

    first = balance_option_positions(source, seed=1002)
    second = balance_option_positions(source, seed=1002)
    different = balance_option_positions(source, seed=1003)

    assert first == second
    assert [question.options for question in first] != [
        question.options for question in different
    ]


def test_prepare_questions_hashes_the_exact_option_presentation() -> None:
    source = _questions()

    first, first_ids, first_options = prepare_questions(
        source, 12, 1002, option_order="balanced", option_seed=7
    )
    second, second_ids, second_options = prepare_questions(
        source, 12, 1002, option_order="balanced", option_seed=8
    )

    assert [question.id for question in first] == [question.id for question in second]
    assert first_ids == second_ids
    assert first_options != second_options


def test_option_presentation_is_stable_when_sample_size_changes() -> None:
    source = _questions(count=20)
    pilot, _, _ = prepare_questions(
        source, 10, 1002, option_order="balanced", option_seed=1002
    )
    full, _, _ = prepare_questions(
        source, 20, 1002, option_order="balanced", option_seed=1002
    )
    full_by_id = {question.id: question for question in full}

    for question in pilot:
        assert question.options == full_by_id[question.id].options
        assert question.gold_index == full_by_id[question.id].gold_index


def test_degenerate_mc_key_is_rejected_before_inference() -> None:
    with pytest.raises(ValueError, match="degenerate answer-position key"):
        validate_mc_option_positions(_questions(count=8))


def test_balanced_mc_key_passes_validation() -> None:
    validate_mc_option_positions(balance_option_positions(_questions(count=8), 1002))
