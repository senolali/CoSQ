"""CoSQ-Graded-Gate: graded self-questioning used only as an answer gate."""

from __future__ import annotations

from cosq import prompts
from cosq.registry import register
from cosq.strategies.cosq_graded import CoSQGradedStrategy


@register("strategy", "cosq_graded_gate")
class CoSQGradedGateStrategy(CoSQGradedStrategy):
    """Graded confidence decides whether to answer; plain CoT generates the answer."""

    name = "cosq_graded_gate"

    def render_answer_prompt(self, question_block: str, certain_needs: list[str]) -> str:
        """Use the byte-identical CoT prompt after the graded gate opens."""
        return prompts.render(
            self.prompt_name("cot"), question_block=question_block, lang=self.lang
        )
