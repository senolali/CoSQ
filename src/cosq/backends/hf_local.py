"""Local Hugging Face backend.

This backend loads a model with ``transformers`` and returns raw generated text.
It is optional because it may require large model weights, a GPU, and additional
packages. Quantization is configurable and recorded in run manifests.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from cosq.backends.base import LLMBackend
from cosq.registry import register
from cosq.types import Completion, GenerationParams

if TYPE_CHECKING:
    from cosq.config import ModelConfig

_IMPORT_HINT = "the 'hf_local' backend needs optional dependencies: pip install 'cosq[hf]'"


@register("backend", "hf_local")
class HFLocalBackend(LLMBackend):
    """`transformers` generation against a locally loaded, pinned model."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        quantization: str | None = "nf4",
        compute_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        if not revision or revision in ("main", "master", "HEAD"):
            raise ValueError(
                f"model revision must be a pinned commit SHA, got {revision!r}. "
                "A floating branch silently changes what a run means."
            )
        if revision.startswith("REPLACE") or revision.startswith("TODO") or "COMMIT" in revision.upper():
            raise ValueError(
                f"revision {revision!r} is still the config template's placeholder. "
                "Set it to the model's commit SHA before running."
            )
        self.model_id = model_id
        self.revision = revision
        self.quantization = quantization
        self.compute_dtype = compute_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self._model: Any = None
        self._tokenizer: Any = None

    @classmethod
    def from_config(cls, config: ModelConfig) -> HFLocalBackend:
        return cls(
            model_id=config.id,
            revision=config.revision,
            quantization=config.quantization,
            compute_dtype=config.compute_dtype,
            **config.backend_params,
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise ImportError(_IMPORT_HINT) from exc

        kwargs: dict[str, Any] = {
            "revision": self.revision,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.quantization == "nf4":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover
                raise ImportError(_IMPORT_HINT) from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, self.compute_dtype),
            )
        elif self.quantization is None:
            kwargs["torch_dtype"] = getattr(torch, self.compute_dtype)
        else:
            raise ValueError(f"unsupported quantization {self.quantization!r}; use 'nf4' or None")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()

    def generate(self, prompt: str, params: GenerationParams) -> Completion:  # pragma: no cover
        import torch

        self._load()
        assert self._tokenizer is not None and self._model is not None
        torch.manual_seed(params.seed)

        messages = [{"role": "user", "content": prompt}]
        if getattr(self._tokenizer, "chat_template", None):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        started = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                do_sample=params.temperature > 0.0,
                temperature=params.temperature if params.temperature > 0.0 else None,
                top_p=params.top_p if params.temperature > 0.0 else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started

        generated = output[0][inputs["input_ids"].shape[-1] :]
        completion = self._tokenizer.decode(generated, skip_special_tokens=True)
        return Completion(
            text=completion,
            prompt=prompt,
            model_id=self.model_id,
            revision=self.revision,
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(generated.shape[-1]),
            latency_s=elapsed,
        )

    def fingerprint(self) -> dict[str, str]:
        return {
            **super().fingerprint(),
            "quantization": str(self.quantization),
            "compute_dtype": self.compute_dtype,
        }
