"""CoSQ-Gate — the self-questioning gate without the constrained generation.

**Why this exists.** The first GPT-4o campaign decomposed CoSQ's coverage-matched
comparison into two effects [M19]:

* **Selection** — the questions CoSQ declares itself certain about really are ones the
  model answers correctly more often. Consistent: 9/9 τ values in the same direction,
  sign test ``p = 0.004``.
* **Generation** — on those same questions, answering "using only the certain
  sub-facts" is no better than plain chain-of-thought. Direction 5/9, ``p = 1.00``:
  noise.

So stages 1-2 produce a real signal and stage 3 spends it. This variant keeps the
signal and drops the spending: identical knowledge-need analysis, identical certainty
assessment, identical decision rule — but when the rule says *answer*, the model gets
the **plain CoT prompt** rather than one constrained to the sub-facts it endorsed.

That makes it a clean ablation. Because the answering prompt is byte-identical to the
``cot`` condition, its accuracy on the questions it answers is CoT's accuracy on those
same questions — which is exactly the quantity ``coverage_matched_risk`` reports. The
contrast CoSQ vs CoSQ-Gate therefore measures the cost of stage 3 alone.
"""

from __future__ import annotations

from cosq import prompts
from cosq.registry import register
from cosq.strategies.cosq import CoSQStrategy


@register("strategy", "cosq_gate")
class CoSQGateStrategy(CoSQStrategy):
    """Self-questioning used purely as a gate on whether to answer."""

    name = "cosq_gate"

    def render_answer_prompt(self, question_block: str, certain_needs: list[str]) -> str:
        """The plain CoT prompt — the endorsed sub-facts are deliberately not passed on.

        Discarding ``certain_needs`` is the entire point of the ablation, not an
        oversight: stages 1-2 decide *whether* to answer, and nothing more.
        """
        return prompts.render(
            self.prompt_name("cot"), question_block=question_block, lang=self.lang
        )
