"""OpenAI API backend.

The backend is intentionally thin: it sends the rendered CoSQ prompt to an OpenAI
chat-completions compatible endpoint and returns the raw text. Parsing, abstention
decisions, scoring, and caching all remain outside the backend.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from cosq.backends.base import LLMBackend
from cosq.registry import register
from cosq.types import Completion, GenerationParams

if TYPE_CHECKING:
    from cosq.config import ModelConfig

_IMPORT_HINT = "the OpenAI backend needs the optional dependency: pip install 'cosq[openai]'"


@register("backend", "openai")
class OpenAIBackend(LLMBackend):
    """Generate completions with the official OpenAI Python SDK.

    Config keys accepted in ``backend_params``:
      - ``token_env``: environment variable containing the API key, default
        ``OPENAI_API_KEY``.
      - ``base_url``: optional OpenAI-compatible base URL.
      - ``system_prompt``: optional system message.
    """

    pinned = False

    def __init__(
        self,
        model_id: str,
        revision: str = "hosted",
        *,
        token: str | None = None,
        token_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        system_prompt: str | None = None,
        extra_model_params: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision or "hosted"
        self.token = token
        self.token_env = token_env
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.extra_model_params = extra_model_params or {}
        self._client: Any = None

    @classmethod
    def from_config(cls, config: ModelConfig) -> OpenAIBackend:
        return cls(model_id=config.id, revision=config.revision, **config.backend_params)

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(_IMPORT_HINT) from exc
        api_key = self.token or os.environ.get(self.token_env)
        if not api_key:
            raise RuntimeError(f"set {self.token_env} or pass token=... in backend_params")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, prompt: str, params: GenerationParams) -> Completion:  # pragma: no cover
        client = self._client_or_create()
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": params.max_new_tokens,
            "temperature": params.temperature,
        }
        if params.temperature > 0:
            payload["top_p"] = params.top_p
        payload.update(self.extra_model_params)
        started = time.perf_counter()
        response = client.chat.completions.create(**payload)
        elapsed = time.perf_counter() - started
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            prompt=prompt,
            model_id=self.model_id,
            revision=self.revision,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_s=elapsed,
        )

    def fingerprint(self) -> dict[str, str]:
        fp = super().fingerprint()
        if self.base_url:
            fp["base_url"] = self.base_url
        return fp
