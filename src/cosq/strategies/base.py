"""The strategy contract."""

from __future__ import annotations

import abc

from cosq import prompts
from cosq.backends.base import LLMBackend
from cosq.types import AnswerRecord, GenerationParams, Question, Turn


def render_question(
    question: Question, lang: str = prompts.DEFAULT_LANG, *, open_ended: bool = False
) -> str:
    """Render a question, with or without its option set.

    Shared by every strategy so the presentation is identical across conditions —
    otherwise the formatting, not the questioning method, would be a second
    manipulated variable.

    ``open_ended`` hides the options entirely. The model must then produce the answer
    from memory rather than recognise it in a list, which is a substantially harder
    task and the one the proposal's mechanism was designed for [M22]. Scoring shifts
    to :func:`~cosq.parsing.match_open_answer` accordingly.
    """
    if open_ended:
        return prompts.render("question_open", question=question.text, lang=lang)
    options = "\n".join(
        f"{chr(ord('A') + index)}) {option}" for index, option in enumerate(question.options)
    )
    return prompts.render("question", question=question.text, options=options, lang=lang)


class Strategy(abc.ABC):
    """Produces one :class:`~cosq.types.AnswerRecord` per question.

    Implementations must append every prompt/completion pair to ``record.trace``
    verbatim. The trace is the evidence; a strategy that discards it makes its runs
    unauditable and un-rescorable.
    """

    name: str

    def __init__(
        self,
        backend: LLMBackend,
        *,
        lang: str = prompts.DEFAULT_LANG,
        open_ended: bool = False,
        mc_output: bool = False,
    ) -> None:
        self.backend = backend
        self.lang = lang
        self.open_ended = open_ended
        self.mc_output = mc_output

    def prompt_name(self, base: str) -> str:
        """Template for ``base`` under the current presentation mode.

        Open-ended variants are separate files rather than a rewrite of the closed-book
        ones, so every earlier run keeps its exact prompts and stays reproducible.
        """
        if self.open_ended:
            return f"{base}_open"
        return f"{base}_mc" if self.mc_output else base

    @abc.abstractmethod
    def answer(self, question: Question, params: GenerationParams, repeat: int = 0) -> AnswerRecord:
        """Answer ``question``, recording the full interrogation trace."""

    def _ask(self, stage: str, prompt: str, params: GenerationParams, record: AnswerRecord) -> str:
        """Call the backend once, appending the exchange to the trace."""
        completion = self.backend.generate(prompt, params)
        record.trace.append(Turn(stage=stage, prompt=prompt, completion=completion.text))
        record.prompt_tokens += completion.prompt_tokens
        record.completion_tokens += completion.completion_tokens
        record.latency_s += completion.latency_s
        record.cost_usd += completion.cost_usd
        return completion.text
