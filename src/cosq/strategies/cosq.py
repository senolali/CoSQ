"""The three-stage Chain-of-Self-Questioning pipeline."""

from __future__ import annotations

from cosq import prompts
from cosq.backends.base import LLMBackend
from cosq.decision.base import DecisionRule
from cosq.decision.threshold import ThresholdRule
from cosq.parsing import parse_certainty, parse_need_list
from cosq.registry import register
from cosq.strategies.base import Strategy, render_question
from cosq.types import AnswerRecord, Certainty, GenerationParams, Question


@register("strategy", "cosq")
class CoSQStrategy(Strategy):
    """Knowledge-need analysis → uncertainty assessment → informed abstention.

    The decision belongs to :class:`~cosq.decision.base.DecisionRule`, not to the
    model: the LLM supplies sub-facts and certainty labels, and the control layer
    decides. Keeping the policy outside the model is what makes τ sweepable offline
    and the mechanism auditable.

    ``unparseable_certainty`` fixes how a stage-2 reply that expresses neither
    certainty nor uncertainty is treated. The default counts it as ``uncertain``:
    a label we could not read is not evidence of knowledge. It is explicit because
    it moves AR, and it is counted in ``record.meta`` so the rate is visible.
    """

    name = "cosq"

    def __init__(
        self,
        backend: LLMBackend,
        *,
        rule: DecisionRule | None = None,
        lang: str = prompts.DEFAULT_LANG,
        open_ended: bool = False,
        unparseable_certainty: Certainty = "uncertain",
    ) -> None:
        super().__init__(backend, lang=lang, open_ended=open_ended)
        self.rule = rule or ThresholdRule(tau=0.0)
        self.unparseable_certainty: Certainty = unparseable_certainty

    def answer(self, question: Question, params: GenerationParams, repeat: int = 0) -> AnswerRecord:
        record = AnswerRecord(
            question_id=question.id,
            strategy=self.name,
            repeat=repeat,
            answer_text="",
            decision="abstain",
        )
        block = render_question(question, self.lang, open_ended=self.open_ended)

        # Stage 1 — knowledge-need analysis.
        needs_text = self._ask(
            "needs",
            prompts.render("cosq_needs", question_block=block, lang=self.lang),
            params,
            record,
        )
        record.needs = parse_need_list(needs_text)

        # Stage 2 — uncertainty assessment, one independent call per sub-fact.
        unreadable = 0
        for index, need in enumerate(record.needs):
            reply = self._ask(
                f"certainty[{index}]",
                prompts.render("cosq_certainty", question_block=block, need=need, lang=self.lang),
                params,
                record,
            )
            parsed = parse_certainty(reply)
            if parsed is None:
                unreadable += 1
                parsed = self.unparseable_certainty
            record.certainties.append(parsed)

        # Stage 3 — the decision, then either abstention or grounded generation.
        outcome = self.rule.decide(record.certainties)
        record.decision = outcome.decision
        record.meta.update(
            uncertain_fraction=outcome.uncertain_fraction,
            n_needs=outcome.n_needs,
            n_uncertain=outcome.n_uncertain,
            decision_reason=outcome.reason,
            rule=repr(self.rule),
            unparseable_certainties=unreadable,
            open_ended=self.open_ended,
        )

        if outcome.decision == "abstain":
            record.answer_text = prompts.load("cosq_abstain", lang=self.lang).strip()
            return record

        certain_needs = [
            need
            for need, certainty in zip(record.needs, record.certainties, strict=True)
            if certainty == "certain"
        ]
        record.answer_text = self._ask(
            "answer",
            self.render_answer_prompt(block, certain_needs),
            params,
            record,
        )
        return record

    def render_answer_prompt(self, question_block: str, certain_needs: list[str]) -> str:
        """Build the stage-3 prompt once the rule has decided to answer.

        Isolated as a hook so a variant can keep the gate (stages 1-2 plus the decision
        rule) while changing what happens after it — see
        :class:`~cosq.strategies.cosq_gate.CoSQGateStrategy`.
        """
        listed = "\n".join(f"- {need}" for need in certain_needs)
        return prompts.render(
            self.prompt_name("cosq_answer"),
            question_block=question_block,
            certain_needs=listed,
            lang=self.lang,
        )
