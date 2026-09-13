"""The three comparison conditions.

``CoTAbstainStrategy`` exists to separate two things the original three-arm design
conflated: the three-stage self-interrogation mechanism, and the mere permission to
say "I don't know". Without it, any HR advantage for CoSQ could be explained by a
one-line prompt change that costs nothing [M3].
"""

from __future__ import annotations

from cosq.parsing import detect_abstention
from cosq.registry import register
from cosq.strategies.base import Strategy, render_question
from cosq.types import AnswerRecord, GenerationParams, Question


class _SinglePrompt(Strategy):
    """One prompt, one completion. Shared by all three baselines."""

    template: str
    allows_abstention: bool

    def answer(self, question: Question, params: GenerationParams, repeat: int = 0) -> AnswerRecord:
        record = AnswerRecord(
            question_id=question.id,
            strategy=self.name,
            repeat=repeat,
            answer_text="",
            decision="answer",
        )
        block = render_question(question, self.lang, open_ended=self.open_ended)
        from cosq import prompts

        prompt = prompts.render(
            self.prompt_name(self.template), question_block=block, lang=self.lang
        )
        text = self._ask("answer", prompt, params, record)
        record.answer_text = text
        if self.allows_abstention and detect_abstention(text):
            record.decision = "abstain"
        record.meta["allows_abstention"] = self.allows_abstention
        record.meta["open_ended"] = self.open_ended
        record.meta["mc_output"] = self.mc_output
        return record


@register("strategy", "direct")
class DirectStrategy(_SinglePrompt):
    """Baseline: answer immediately, no reasoning, no abstention offered."""

    name = "direct"
    template = "direct"
    allows_abstention = False


@register("strategy", "cot")
class CoTStrategy(_SinglePrompt):
    """Baseline: chain-of-thought. The pre-registered comparator for H1."""

    name = "cot"
    template = "cot"
    allows_abstention = False


@register("strategy", "cot_abstain")
class CoTAbstainStrategy(_SinglePrompt):
    """Chain-of-thought *plus* explicit permission to abstain.

    The comparator that isolates what CoSQ's three stages add over the free
    alternative of simply allowing "I don't know" [M3].
    """

    name = "cot_abstain"
    template = "cot_abstain"
    allows_abstention = True
