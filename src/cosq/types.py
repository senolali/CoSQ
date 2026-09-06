"""Core data types.

These are the contracts every subsystem speaks. Keeping them here — rather than
letting each module define its own — is what makes backends, strategies, parsers,
and scorers independently replaceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

#: Outcome of scoring a single answer. ``unparseable`` is deliberately distinct
#: from ``wrong``: an answer the parser could not map to an option is a *measurement*
#: failure, not a hallucination, and folding the two together inflates HR [M9].
Outcome = Literal["correct", "wrong", "idk", "unparseable"]

#: The model's self-assessed certainty about one sub-fact (CoSQ stage 2).
Certainty = Literal["certain", "uncertain"]

#: What the decision rule concluded (CoSQ stage 3).
Decision = Literal["answer", "abstain"]

OUTCOMES: tuple[Outcome, ...] = ("correct", "wrong", "idk", "unparseable")


@dataclass(frozen=True, slots=True)
class Question:
    """One benchmark item.

    ``options`` is the multiple-choice option set and ``gold_index`` points at the
    single correct option (TruthfulQA MC1). Both are required: this project scores
    by generation-then-match against the option set, never by log-likelihood [M9].
    """

    id: str
    text: str
    options: tuple[str, ...]
    gold_index: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(f"question {self.id!r} has no options")
        if not 0 <= self.gold_index < len(self.options):
            raise ValueError(
                f"question {self.id!r}: gold_index {self.gold_index} out of range "
                f"for {len(self.options)} options"
            )

    @property
    def gold(self) -> str:
        return self.options[self.gold_index]


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Decoding parameters. Identical across every condition — the questioning
    method is the only manipulated variable."""

    temperature: float = 0.0
    top_p: float = 0.95
    max_new_tokens: int = 256
    seed: int = 1002

    def cache_key(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class Completion:
    """Raw model output. Backends return this and never interpret it."""

    text: str
    prompt: str
    model_id: str
    revision: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    #: Billed cost of this call, when the provider reports one. Hosted inference is
    #: metered, so cost is part of a run's provenance, not an afterthought.
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class Turn:
    """One prompt/completion exchange inside a strategy, kept verbatim.

    The full list of turns is the audit trail: it is what lets a run be re-scored
    offline, and what a reviewer inspects to see what the model actually said.
    """

    stage: str
    prompt: str
    completion: str


@dataclass(slots=True)
class AnswerRecord:
    """What a strategy produced for one question on one repeat.

    This is written to ``records.jsonl`` before any parsing or scoring happens.
    """

    question_id: str
    strategy: str
    repeat: int
    answer_text: str
    decision: Decision
    trace: list[Turn] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    certainties: list[Certainty] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.decision == "abstain"

    @property
    def n_needs(self) -> int:
        """Number of sub-facts stage 1 enumerated.

        Recorded per question because under a strict rule the abstention
        probability grows with it, which would otherwise confound AR [M7].
        """
        return len(self.needs)


@dataclass(slots=True)
class ScoredRecord:
    """An :class:`AnswerRecord` plus its outcome. Produced by the scorer, offline."""

    record: AnswerRecord
    outcome: Outcome
    matched_index: int | None = None

    @property
    def question_id(self) -> str:
        return self.record.question_id

    @property
    def strategy(self) -> str:
        return self.record.strategy

    @property
    def repeat(self) -> int:
        return self.record.repeat

    @property
    def answered(self) -> bool:
        """Did the system commit to an answer? ``unparseable`` counts as answered —
        the model did produce something, we just could not map it."""
        return self.outcome in ("correct", "wrong", "unparseable")

    def with_outcome(self, outcome: Outcome) -> ScoredRecord:
        return replace(self, outcome=outcome)
