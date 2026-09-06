"""A deterministic offline backend.

Every test in this repository runs against this backend: no network, no GPU, no
model weights. It is a *pseudo-model*, not a simulation of any real one — it exists
to exercise the pipeline's mechanics (prompting, parsing, decision, scoring,
caching), never to produce numbers that mean anything.

Determinism: the reply is a pure function of ``(prompt, params)``, seeded by their
hash. The same prompt always yields the same text, which is what makes the cache and
the resumption logic testable.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import TYPE_CHECKING

from cosq.backends.base import LLMBackend
from cosq.registry import register
from cosq.types import Completion, GenerationParams

if TYPE_CHECKING:
    from cosq.config import ModelConfig

_OPTION_LINE = re.compile(r"^\s*([A-Z])\)\s+(.*\S)\s*$", re.MULTILINE)

_STAGE_CUES = (
    ("needs", "Required information:"),
    ("certainty", "Are you certain of this information?"),
    ("confidence", "How confident are you"),
)


@register("backend", "mock")
class MockBackend(LLMBackend):
    """Deterministic pseudo-model.

    Args:
        accuracy: probability of picking the correct-looking option. The mock cannot
            know which option is gold — it picks uniformly and lets the scorer decide
            — so this only shifts how often it picks *the first* option.
        p_uncertain: probability that a binary stage-2 reply is "Uncertain". Graded
            replies draw a percentage from ``confidence_range`` instead.
        confidence_range: inclusive bounds for a graded stage-2 reply, as percentages.
            The default straddles a typical 0.8 threshold so a smoke run exercises both
            branches of the decision rather than abstaining on everything.
        p_abstain: probability that a free-form answer is an explicit refusal.
        n_needs: inclusive range for how many sub-facts stage 1 invents.
        responses: substring -> canned reply. Checked first, so a test can pin the
            exact text a given prompt returns.
    """

    def __init__(
        self,
        *,
        accuracy: float = 0.5,
        p_uncertain: float = 0.3,
        p_abstain: float = 0.05,
        n_needs: tuple[int, int] = (2, 4),
        confidence_range: tuple[int, int] = (55, 98),
        responses: dict[str, str] | None = None,
        model_id: str = "mock",
        revision: str = "n/a",
    ) -> None:
        self.accuracy = accuracy
        self.p_uncertain = p_uncertain
        self.p_abstain = p_abstain
        self.n_needs = n_needs
        self.confidence_range = confidence_range
        self.responses = responses or {}
        self.model_id = model_id
        self.revision = revision
        self.calls = 0

    @classmethod
    def from_config(cls, config: ModelConfig) -> MockBackend:
        return cls(model_id=config.id, revision=config.revision, **config.backend_params)

    def _rng(self, prompt: str, params: GenerationParams) -> random.Random:
        payload = json.dumps({"p": prompt, "g": params.cache_key()}, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    @staticmethod
    def _stage(prompt: str) -> str:
        for stage, cue in _STAGE_CUES:
            if cue in prompt:
                return stage
        return "answer"

    @staticmethod
    def _options(prompt: str) -> list[tuple[str, str]]:
        return [(m.group(1), m.group(2)) for m in _OPTION_LINE.finditer(prompt)]

    def generate(self, prompt: str, params: GenerationParams) -> Completion:
        self.calls += 1
        for needle, canned in self.responses.items():
            if needle in prompt:
                return self._completion(canned, prompt)

        rng = self._rng(prompt, params)
        stage = self._stage(prompt)
        if stage == "needs":
            count = rng.randint(*self.n_needs)
            body = "\n".join(
                f"{i}. sub-fact {i} required by this question" for i in range(1, count + 1)
            )
            return self._completion(body, prompt)
        if stage == "certainty":
            return self._completion(
                "Uncertain" if rng.random() < self.p_uncertain else "Certain", prompt
            )
        if stage == "confidence":
            return self._completion(str(rng.randint(*self.confidence_range)), prompt)

        if rng.random() < self.p_abstain:
            return self._completion("I don't know.", prompt)
        options = self._options(prompt)
        if not options:
            # Open-ended presentation: the options are not in the prompt, so a
            # deterministic stand-in cannot know the answer. It emits a well-formed
            # sentence so the plumbing is exercised, and the scorer correctly calls it
            # `unparseable`. A mock open-ended run is a pipeline check, not a
            # behavioural one [M22].
            return self._completion(
                "Answer: this is a placeholder free-form answer from the mock backend.",
                prompt,
            )
        label, _ = options[0] if rng.random() < self.accuracy else rng.choice(options)
        reasoning = "Reasoning: weighing the options against the stated sub-facts.\n"
        return self._completion(f"{reasoning}Answer: {label}", prompt)

    def _completion(self, text: str, prompt: str) -> Completion:
        return Completion(
            text=text,
            prompt=prompt,
            model_id=self.model_id,
            revision=self.revision,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            latency_s=0.0,
        )
