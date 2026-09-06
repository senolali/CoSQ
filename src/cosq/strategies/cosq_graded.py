"""CoSQ-Graded — stage 2 asks *how* confident, not merely whether.

Stage 1 and stage 3 are unchanged. What changes is the decision variable: instead of a
binary "Certain / Uncertain" per sub-fact collapsed by a conjunction, each sub-fact
gets a confidence in ``[0, 1]`` and an explicit aggregator combines them into a
continuous score [M20].

This is a strictly more expressive version of the same idea, not a different one:
:class:`~cosq.decision.confidence.ConfidenceRule` reproduces the pre-registered τ rule
exactly on binary inputs, so the original remains a special case.
"""

from __future__ import annotations

from cosq import prompts
from cosq.backends.base import LLMBackend
from cosq.decision.confidence import ConfidenceRule
from cosq.parsing.stages import parse_confidence
from cosq.registry import register
from cosq.strategies.base import render_question
from cosq.strategies.cosq import CoSQStrategy
from cosq.types import AnswerRecord, GenerationParams, Question


@register("strategy", "cosq_graded")
class CoSQGradedStrategy(CoSQStrategy):
    """Three stages, with a graded confidence signal in the middle.

    ``unparseable_confidence`` fixes what an unreadable stage-2 reply is worth. The
    default is ``0.0`` — a confidence we could not read is not evidence of knowledge,
    matching the binary variant's choice to treat it as ``uncertain``. It is explicit
    because it moves AR, and the count is recorded so the rate stays visible.
    """

    name = "cosq_graded"

    def __init__(
        self,
        backend: LLMBackend,
        *,
        rule: ConfidenceRule | None = None,
        lang: str = prompts.DEFAULT_LANG,
        open_ended: bool = False,
        unparseable_confidence: float = 0.0,
    ) -> None:
        # The parent stores a binary rule; this variant never consults it.
        super().__init__(backend, lang=lang, open_ended=open_ended)
        self.confidence_rule = rule or ConfidenceRule(threshold=0.8, aggregator="minimum")
        if not 0.0 <= unparseable_confidence <= 1.0:
            raise ValueError(
                f"unparseable_confidence must be in [0, 1], got {unparseable_confidence}"
            )
        self.unparseable_confidence = unparseable_confidence

    def answer(self, question: Question, params: GenerationParams, repeat: int = 0) -> AnswerRecord:
        record = AnswerRecord(
            question_id=question.id,
            strategy=self.name,
            repeat=repeat,
            answer_text="",
            decision="abstain",
        )
        block = render_question(question, self.lang, open_ended=self.open_ended)

        # Stage 1 — unchanged.
        needs_text = self._ask(
            "needs",
            prompts.render("cosq_needs", question_block=block, lang=self.lang),
            params,
            record,
        )
        from cosq.parsing import parse_need_list

        record.needs = parse_need_list(needs_text)

        # Stage 2 — graded rather than binary.
        confidences: list[float] = []
        unreadable = 0
        for index, need in enumerate(record.needs):
            reply = self._ask(
                f"confidence[{index}]",
                prompts.render("cosq_confidence", question_block=block, need=need, lang=self.lang),
                params,
                record,
            )
            value = parse_confidence(reply)
            if value is None:
                unreadable += 1
                value = self.unparseable_confidence
            confidences.append(value)
            # Kept alongside so both variants' records carry comparable fields.
            record.certainties.append(
                "certain" if value >= self.confidence_rule.threshold else "uncertain"
            )

        outcome = self.confidence_rule.decide(confidences)
        record.decision = outcome.decision
        record.meta.update(
            confidences=confidences,
            aggregate_confidence=outcome.score,
            uncertain_fraction=outcome.uncertain_fraction,
            n_needs=outcome.n_needs,
            n_uncertain=outcome.n_uncertain,
            decision_reason=outcome.reason,
            rule=repr(self.confidence_rule),
            unparseable_confidences=unreadable,
            open_ended=self.open_ended,
        )

        if outcome.decision == "abstain":
            record.answer_text = prompts.load("cosq_abstain", lang=self.lang).strip()
            return record

        certain_needs = [
            need
            for need, value in zip(record.needs, confidences, strict=True)
            if value >= self.confidence_rule.threshold
        ]
        record.answer_text = self._ask(
            "answer", self.render_answer_prompt(block, certain_needs), params, record
        )
        return record
