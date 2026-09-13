"""Adaptive grounded CoSQ with critical facts and a contradiction check."""

from __future__ import annotations

import re

from cosq import prompts
from cosq.backends.base import LLMBackend
from cosq.decision.confidence import ConfidenceRule
from cosq.parsing import parse_need_list
from cosq.parsing.stages import parse_confidence
from cosq.registry import register
from cosq.strategies.base import render_question
from cosq.types import AnswerRecord, GenerationParams, Question

_FIELD = re.compile(r"^\s*(ROLE|FACT|CONFIDENCE)\s*:\s*(.*?)\s*$", re.I | re.M)


@register("strategy", "cosq_grounded_adaptive")
class CoSQGroundedAdaptiveStrategy:
    """Use role-aware factual claims, adaptive answering, and contradiction checking."""

    name = "cosq_grounded_adaptive"

    def __init__(
        self,
        backend: LLMBackend,
        *,
        rule: ConfidenceRule | None = None,
        lang: str = prompts.DEFAULT_LANG,
        open_ended: bool = False,
        mc_output: bool = False,
        critical_threshold: float = 0.65,
        overall_threshold: float = 0.60,
        minimum_critical: float = 0.40,
        confident_threshold: float = 0.75,
        unparseable_confidence: float = 0.0,
    ) -> None:
        self.backend = backend
        self.lang = lang
        self.open_ended = open_ended
        self.mc_output = mc_output
        if rule is not None:
            overall_threshold = rule.threshold
        self.critical_threshold = critical_threshold
        self.overall_threshold = overall_threshold
        self.minimum_critical = minimum_critical
        self.confident_threshold = confident_threshold
        self.unparseable_confidence = unparseable_confidence

    def _ask(self, stage: str, prompt: str, params: GenerationParams, record: AnswerRecord) -> str:
        completion = self.backend.generate(prompt, params)
        from cosq.types import Turn

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
        needs = self._ask(
            "needs",
            prompts.render("cosq_needs", question_block=block, lang=self.lang),
            params,
            record,
        )
        record.needs = parse_need_list(needs)
        claims: list[dict[str, object]] = []
        for index, need in enumerate(record.needs):
            reply = self._ask(
                f"fact_confidence[{index}]",
                prompts.render(
                    "cosq_fact_role_confidence", question_block=block, need=need, lang=self.lang
                ),
                params,
                record,
            )
            fields = {key.upper(): value.strip() for key, value in _FIELD.findall(reply)}
            role = (
                "critical"
                if fields.get("ROLE", "").lower().startswith("critical")
                else "supporting"
            )
            fact = fields.get("FACT", "")
            confidence = parse_confidence(fields.get("CONFIDENCE", reply))
            if confidence is None:
                confidence = self.unparseable_confidence
            claims.append({"role": role, "fact": fact, "confidence": confidence})
            record.certainties.append(
                "certain" if confidence >= self.critical_threshold else "uncertain"
            )

        values = [float(c["confidence"]) for c in claims]
        critical = [float(c["confidence"]) for c in claims if c["role"] == "critical"]
        overall = sum(values) / len(values) if values else 0.0
        critical_mean = sum(critical) / len(critical) if critical else 0.0
        critical_min = min(critical) if critical else 0.0
        passes = (
            bool(values)
            and overall >= self.overall_threshold
            and critical_mean >= self.critical_threshold
            and critical_min >= self.minimum_critical
        )
        mode = "confident" if overall >= self.confident_threshold else "cautious"
        record.meta.update(
            claims=claims,
            aggregate_confidence=overall,
            critical_mean=critical_mean,
            critical_min=critical_min,
            decision_reason=(
                f"overall={overall:.3f}, critical_mean={critical_mean:.3f}, "
                f"critical_min={critical_min:.3f}"
            ),
            thresholds={
                "overall": self.overall_threshold,
                "critical_mean": self.critical_threshold,
                "critical_min": self.minimum_critical,
            },
            response_mode=mode,
            open_ended=self.open_ended,
        )
        if not passes:
            record.answer_text = prompts.load("cosq_abstain", lang=self.lang).strip()
            return record

        accepted = "\n".join(
            f"- {c['fact']}"
            for c in claims
            if c["fact"] and float(c["confidence"]) >= self.minimum_critical
        )
        answer = self._ask(
            "answer",
            prompts.render(
                "cosq_grounded_adaptive_answer_mc"
                if self.mc_output
                else "cosq_grounded_adaptive_answer",
                question_block=block,
                accepted_facts=accepted,
                response_mode=mode,
                lang=self.lang,
            ),
            params,
            record,
        )
        check = self._ask(
            "consistency_check",
            prompts.render(
                "cosq_consistency_check",
                question_block=block,
                accepted_facts=accepted,
                answer=answer,
                lang=self.lang,
            ),
            params,
            record,
        )
        if re.search(r"\bCONTRADICTORY\b", check, re.I):
            record.decision = "abstain"
            record.answer_text = prompts.load("cosq_abstain", lang=self.lang).strip()
            record.meta["consistency"] = "contradictory"
            return record
        record.decision = "answer"
        record.answer_text = answer
        record.meta["consistency"] = "consistent_or_unresolved"
        return record
