"""CoSQ-Grounded: self-generated facts plus confidence before answering."""

from __future__ import annotations

import re

from cosq import prompts
from cosq.backends.base import LLMBackend
from cosq.decision.confidence import ConfidenceRule
from cosq.parsing import parse_need_list
from cosq.parsing.stages import parse_confidence
from cosq.registry import register
from cosq.strategies.base import render_question
from cosq.strategies.cosq_graded import CoSQGradedStrategy
from cosq.types import AnswerRecord, GenerationParams, Question

_FACT = re.compile(r"^\s*(?:FACT|CLAIM)\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)


@register("strategy", "cosq_grounded")
class CoSQGroundedStrategy(CoSQGradedStrategy):
    """Generate a factual sub-claim, confidence, and then answer from accepted claims."""

    name = "cosq_grounded"

    def __init__(
        self,
        backend: LLMBackend,
        *,
        rule: ConfidenceRule | None = None,
        lang: str = prompts.DEFAULT_LANG,
        open_ended: bool = False,
        mc_output: bool = False,
        unparseable_confidence: float = 0.0,
    ) -> None:
        super().__init__(
            backend,
            rule=rule,
            lang=lang,
            open_ended=open_ended,
            mc_output=mc_output,
            unparseable_confidence=unparseable_confidence,
        )

    def answer(self, question: Question, params: GenerationParams, repeat: int = 0) -> AnswerRecord:
        record = AnswerRecord(
            question_id=question.id,
            strategy=self.name,
            repeat=repeat,
            answer_text="",
            decision="abstain",
        )
        block = render_question(question, self.lang, open_ended=self.open_ended)
        needs_text = self._ask(
            "needs",
            prompts.render("cosq_needs", question_block=block, lang=self.lang),
            params,
            record,
        )
        record.needs = parse_need_list(needs_text)

        facts: list[str] = []
        confidences: list[float] = []
        unreadable = 0
        for index, need in enumerate(record.needs):
            reply = self._ask(
                f"fact_confidence[{index}]",
                prompts.render(
                    "cosq_fact_confidence", question_block=block, need=need, lang=self.lang
                ),
                params,
                record,
            )
            fact_match = _FACT.search(reply)
            facts.append(fact_match.group(1).strip() if fact_match else "")
            value = parse_confidence(reply)
            if value is None:
                unreadable += 1
                value = self.unparseable_confidence
            confidences.append(value)
            record.certainties.append(
                "certain" if value >= self.confidence_rule.threshold else "uncertain"
            )

        outcome = (
            self.confidence_rule.decide(confidences)
            if confidences
            else self.confidence_rule.decide([])
        )
        record.decision = outcome.decision
        accepted = [
            fact
            for fact, value in zip(facts, confidences, strict=True)
            if fact and value >= self.confidence_rule.threshold
        ]
        record.meta.update(
            facts=facts,
            accepted_facts=accepted,
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

        fact_block = "\n".join(f"- {fact}" for fact in accepted)
        record.answer_text = self._ask(
            "answer",
            prompts.render(
                "cosq_grounded_answer_mc" if self.mc_output else "cosq_grounded_answer",
                question_block=block,
                accepted_facts=fact_block,
                lang=self.lang,
            ),
            params,
            record,
        )
        return record
