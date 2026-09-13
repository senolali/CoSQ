"""Typed configuration and the run identity that follows from it.

An experiment is a pure function of its config, so the config *is* the run's
identity: :func:`config_hash` folds in the prompt templates, which means editing a
prompt produces a different hash and therefore a different run. That is deliberate —
a prompt change is a protocol change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from cosq import prompts
from cosq.types import GenerationParams

if TYPE_CHECKING:
    from cosq.backends.base import LLMBackend
    from cosq.data.base import DatasetAdapter
    from cosq.strategies.base import Strategy


def load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def _known_fields(cls: type, mapping: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return mapping


@dataclass(frozen=True, slots=True)
class ModelConfig:
    id: str
    revision: str
    backend: str = "hf_local"
    quantization: str | None = "nf4"
    compute_dtype: str = "bfloat16"
    generation: dict[str, Any] = field(default_factory=dict)
    #: Backend-specific construction arguments, passed through verbatim. Keeps
    #: provider quirks out of the shared schema.
    backend_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> ModelConfig:
        return cls(**_known_fields(cls, dict(mapping)))

    def params(self) -> GenerationParams:
        return GenerationParams(**self.generation) if self.generation else GenerationParams()

    def build(self) -> LLMBackend:
        """Instantiate the backend named by ``backend``.

        Each backend knows how to read its own configuration, so adding a provider
        never means editing this module.
        """
        from cosq.registry import resolve

        backend_cls = cast("type[LLMBackend]", resolve("backend", self.backend))
        return backend_cls.from_config(self)


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    name: str
    tau: float = 0.0
    on_empty: str = "abstain"
    #: Graded variants only: the aggregate-confidence cutoff and how per-sub-fact
    #: confidences are combined. See cosq.decision.confidence.AGGREGATORS.
    threshold: float = 0.8
    aggregator: str = "minimum"
    #: Name this configuration appears under in records and metrics. Defaults to
    #: ``name``; set it when one experiment runs the same strategy twice with
    #: different settings, so the two do not merge into one condition.
    label: str | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> StrategyConfig:
        return cls(**_known_fields(cls, dict(mapping)))

    @property
    def condition(self) -> str:
        """The label this configuration is reported under."""
        return self.label or self.name

    def build(
        self, backend: LLMBackend, *, open_ended: bool = False, mc_output: bool = False
    ) -> Strategy:
        from cosq.decision.threshold import EmptyPolicy, ThresholdRule
        from cosq.registry import resolve

        strategy_cls = resolve("strategy", self.name)
        if self.name in (
            "cosq_graded",
            "cosq_graded_gate",
            "cosq_grounded",
            "cosq_grounded_adaptive",
            "cosq_critical_grounded",
        ):
            from cosq.decision.confidence import ConfidenceRule

            graded = ConfidenceRule(
                threshold=self.threshold, aggregator=self.aggregator, on_empty=self.on_empty
            )
            kwargs = {"rule": graded, "open_ended": open_ended}
            if self.name in ("cosq_grounded", "cosq_grounded_adaptive", "cosq_critical_grounded"):
                kwargs["mc_output"] = mc_output
            return cast("Strategy", strategy_cls(backend, **kwargs))
        if self.name in ("cosq", "cosq_gate"):
            rule = ThresholdRule(tau=self.tau, on_empty=cast("EmptyPolicy", self.on_empty))
            return cast("Strategy", strategy_cls(backend, rule=rule, open_ended=open_ended))
        return cast("Strategy", strategy_cls(backend, open_ended=open_ended, mc_output=mc_output))


@dataclass(frozen=True, slots=True)
class DataConfig:
    name: str = "truthfulqa_mc1"
    n: int = 100
    seed: int = 1002
    path: str | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> DataConfig:
        return cls(**_known_fields(cls, dict(mapping)))

    def build(self) -> DatasetAdapter:
        from cosq.registry import resolve

        dataset_cls = resolve("dataset", self.name)
        built = dataset_cls(self.path) if self.path else dataset_cls()
        return cast("DatasetAdapter", built)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """A complete, self-describing experiment."""

    name: str
    model: ModelConfig
    data: DataConfig
    strategies: tuple[StrategyConfig, ...]
    repeats: int = 3
    seed: int = 1002
    lang: str = "en"
    #: Hide the option set from the model, so it answers from memory instead of
    #: recognising the answer in a list. Applies to every condition uniformly — a grid
    #: mixing the two presentations would confound condition with task difficulty, so
    #: it is not expressible [M22].
    open_ended: bool = False
    mc_output: bool = False

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> ExperimentConfig:
        payload = dict(mapping)
        model = payload.pop("model")
        data = payload.pop("data", {})
        strategies = payload.pop("strategies")
        if isinstance(model, str):
            model = load_yaml(model)
        built = tuple(
            StrategyConfig.from_mapping({"name": s} if isinstance(s, str) else s)
            for s in strategies
        )
        # Two configurations sharing a condition name would be merged into one
        # condition by the scorer — silently, and with no way to tell afterwards.
        seen = [s.condition for s in built]
        duplicates = sorted({c for c in seen if seen.count(c) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate condition name(s) {duplicates}: two strategies would be "
                "recorded under the same label and merged. Give each a distinct `label`."
            )
        return cls(
            model=ModelConfig.from_mapping(model),
            data=DataConfig.from_mapping(data),
            strategies=built,
            **_known_fields(cls, payload),
        )

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        return cls.from_mapping(load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def total_queries(self) -> int:
        """Workload, not sample size. The statistical unit is the question [M13]."""
        return self.data.n * len(self.strategies) * self.repeats


def config_hash(config: ExperimentConfig, *, length: int = 12) -> str:
    """Stable short hash over the resolved config *and* the prompt templates."""
    payload = json.dumps(
        {"config": config.to_dict(), "prompts": prompts.prompt_digest(config.lang)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
