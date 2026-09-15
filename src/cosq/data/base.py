"""The dataset contract and seeded sampling."""

from __future__ import annotations

import abc
import hashlib
import random
from collections.abc import Sequence
from dataclasses import replace

from cosq.types import Question


class DatasetAdapter(abc.ABC):
    """Yields :class:`~cosq.types.Question` objects with gold answers and options."""

    name: str

    @abc.abstractmethod
    def load(self) -> list[Question]:
        """Return every available question, in a stable order."""


def sample_questions(
    questions: Sequence[Question], n: int, seed: int
) -> tuple[list[Question], str]:
    """Draw ``n`` questions reproducibly.

    Sorting by id before sampling makes the draw independent of the order the source
    happened to yield, so the same seed gives the same subset across machines and
    library versions. Returns the sample and a SHA-256 over its ids, which is written
    into the run manifest so the exact subset is recoverable later.
    """
    if n > len(questions):
        raise ValueError(f"asked for {n} questions but only {len(questions)} are available")
    ordered = sorted(questions, key=lambda question: question.id)
    sample = random.Random(seed).sample(ordered, n) if n < len(ordered) else list(ordered)
    sample.sort(key=lambda question: question.id)

    digest = hashlib.sha256()
    for question in sample:
        digest.update(question.id.encode("utf-8"))
        digest.update(b"\x00")
    return sample, digest.hexdigest()


def _stable_digest(seed: int, *parts: object) -> bytes:
    payload = "\x00".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def balance_option_positions(questions: Sequence[Question], seed: int) -> list[Question]:
    """Reorder options reproducibly while balancing the correct label.

    Correct positions are balanced separately within each option-count group.
    This prevents a fixed source position, such as an always-first gold answer,
    from turning option-label preference into benchmark performance. Distractors
    are also deterministically permuted.
    """
    by_option_count: dict[int, list[Question]] = {}
    for question in questions:
        by_option_count.setdefault(len(question.options), []).append(question)

    transformed: dict[str, Question] = {}
    for option_count, group in sorted(by_option_count.items()):
        ordered = sorted(
            group,
            key=lambda question: _stable_digest(
                seed, "correct-position", option_count, question.id
            ),
        )
        offset = int.from_bytes(
            _stable_digest(seed, "position-offset", option_count)[:8], "big"
        ) % option_count
        for rank, question in enumerate(ordered):
            target_position = (offset + rank) % option_count
            distractors = [
                index for index in range(option_count) if index != question.gold_index
            ]
            distractors.sort(
                key=lambda index: _stable_digest(seed, "distractor", question.id, index)
            )
            permutation = list(distractors)
            permutation.insert(target_position, question.gold_index)
            meta = dict(question.meta)
            meta.update(
                {
                    "option_order": "balanced",
                    "option_seed": seed,
                    "source_gold_index": question.gold_index,
                    # New option index -> source option index.
                    "option_permutation": permutation,
                }
            )
            transformed[question.id] = replace(
                question,
                options=tuple(question.options[index] for index in permutation),
                gold_index=target_position,
                meta=meta,
            )

    return [transformed[question.id] for question in questions]


def options_digest(questions: Sequence[Question]) -> str:
    """Hash the exact option presentation and answer key used by a run."""
    digest = hashlib.sha256()
    for question in questions:
        digest.update(question.id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(question.gold_index).encode("ascii"))
        digest.update(b"\x00")
        for option in question.options:
            digest.update(option.encode("utf-8"))
            digest.update(b"\x00")
    return digest.hexdigest()


def prepare_questions(
    questions: Sequence[Question],
    n: int,
    sample_seed: int,
    *,
    option_order: str = "source",
    option_seed: int | None = None,
) -> tuple[list[Question], str, str]:
    """Sample questions and apply the configured option presentation."""
    if option_order == "balanced":
        presented = balance_option_positions(
            questions, sample_seed if option_seed is None else option_seed
        )
    elif option_order == "source":
        presented = list(questions)
    else:
        raise ValueError(
            f"unknown option_order {option_order!r}; expected 'source' or 'balanced'"
        )
    sample, ids_sha256 = sample_questions(presented, n, sample_seed)
    return sample, ids_sha256, options_digest(sample)


def validate_mc_option_positions(
    questions: Sequence[Question], *, minimum_size: int = 8
) -> None:
    """Reject a degenerate generative-MC key before inference starts."""
    eligible = [question for question in questions if len(question.options) > 1]
    positions = {question.gold_index for question in eligible}
    if len(eligible) >= minimum_size and len(positions) == 1:
        only = next(iter(positions))
        label = chr(ord("A") + only) if only < 26 else str(only)
        raise ValueError(
            "generative multiple-choice evaluation has a degenerate answer-position "
            f"key: all {len(eligible)} correct answers are at position {label}. "
            "Set data.option_order: balanced so option-label preference cannot solve "
            "the benchmark."
        )
