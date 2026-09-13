"""Critical-Only Grounded CoSQ."""

from __future__ import annotations

import re

from cosq import prompts
from cosq.backends.base import LLMBackend
from cosq.decision.confidence import ConfidenceRule
from cosq.parsing.stages import parse_confidence
from cosq.registry import register
from cosq.strategies.base import render_question
from cosq.types import AnswerRecord, GenerationParams, Question, Turn

_ITEM = re.compile(
    r"^\s*(?:(?:[-*]|\(?\d{1,2}[.)])\s*)?\[(CRITICAL|SUPPORTING)\]\s*(.+?)\s*$",
    re.IGNORECASE,
)


def _parse_role_items(text: str) -> tuple[list[str], list[str]]:
    critical: list[str] = []
    supporting: list[str] = []
    for raw in text.splitlines():
        match = _ITEM.match(raw.strip())
        if not match:
            continue
        role, item = match.group(1).lower(), match.group(2).strip()
        if not item:
            continue
        target = critical if role == "critical" else supporting
        if item.casefold() not in {x.casefold() for x in target}:
            target.append(item)
    return critical, supporting


@register("strategy", "cosq_critical_grounded")
class CoSQCriticalGroundedStrategy:
    """Evaluate only model-labelled critical information needs with a graded gate."""

    name = "cosq_critical_grounded"

    def __init__(
        self,
        backend: LLMBackend,
        *,
        rule: ConfidenceRule | None = None,
        lang: str = prompts.DEFAULT_LANG,
        open_ended: bool = False,
        mc_output: bool = False,
        minimum_fact_confidence: float = 0.40,
        unparseable_confidence: float = 0.0,
    ) -> None:
        self.backend = backend
        self.lang = lang
        self.open_ended = open_ended
        self.mc_output = mc_output
        self.rule = rule or ConfidenceRule(threshold=0.7, aggregator="mean")
        self.minimum_fact_confidence = minimum_fact_confidence
        self.unparseable_confidence = unparseable_confidence

    def _ask(self, stage: str, prompt: str, params: GenerationParams, record: AnswerRecord) -> str:
        completion = self.backend.generate(prompt, params)
        record.trace.append(Turn(stage=stage, prompt=prompt, completion=completion.text))
        record.prompt_tokens += completion.prompt_tokens
        record.completion_tokens += completion.completion_tokens
        record.latency_s += completion.latency_s
        record.cost_usd += completion.cost_usd
        return completion.text

    def answer(self, question: Question, params: GenerationParams, repeat: int = 0) -> AnswerRecord:
        record = AnswerRecord(
            question_id=question.id,
            strategy=self.name,
            repeat=repeat,
            answer_text="",
            decision="abstain",
        )
        block = render_question(question, self.lang, open_ended=self.open_ended)
        needs_reply = self._ask(
            "needs",
            prompts.render("cosq_needs_roles", question_block=block, lang=self.lang),
            params,
            record,
        )
        critical, supporting = _parse_role_items(needs_reply)
        confidences: list[float] = []
        facts: list[str] = []
        unreadable = 0
        for index, need in enumerate(critical):
            reply = self._ask(
                f"critical_fact_confidence[{index}]",
                prompts.render(
                    "cosq_critical_fact_confidence",
                    question_block=block,
                    need=need,
                    lang=self.lang,
                ),
                params,
                record,
            )
            fact_match = re.search(r"^\s*FACT\s*:\s*(.*?)\s*$", reply, re.I | re.M)
            facts.append(fact_match.group(1).strip() if fact_match else "")
            confidence_match = re.search(r"^\s*CONFIDENCE\s*:\s*(.*?)\s*$", reply, re.I | re.M)
            value = parse_confidence(confidence_match.group(1) if confidence_match else reply)
            if value is None:
                unreadable += 1
                value = self.unparseable_confidence
            confidences.append(value)

        outcome = self.rule.decide(confidences) if confidences else self.rule.decide([])
        accepted = [
            fact
            for fact, value in zip(facts, confidences, strict=True)
            if fact and value >= self.minimum_fact_confidence
        ]
        record.decision = outcome.decision
        record.meta.update(
            critical_needs=critical,
            supporting_needs=supporting,
            facts=facts,
            confidences=confidences,
            accepted_facts=accepted,
            aggregate_confidence=outcome.score,
            n_critical=len(critical),
            n_supporting=len(supporting),
            decision_reason=outcome.reason,
            rule=repr(self.rule),
            unparseable_confidences=unreadable,
            open_ended=self.open_ended,
            mc_output=self.mc_output,
        )
        if outcome.decision == "abstain":
            record.answer_text = prompts.load("cosq_abstain", lang=self.lang).strip()
            return record

        fact_block = "\n".join(f"- {fact}" for fact in accepted)
        record.answer_text = self._ask(
            "answer",
            prompts.render(
                "cosq_critical_grounded_answer_mc"
                if self.mc_output
                else "cosq_critical_grounded_answer",
                question_block=block,
                accepted_facts=fact_block,
                lang=self.lang,
            ),
            params,
            record,
        )
        return record
