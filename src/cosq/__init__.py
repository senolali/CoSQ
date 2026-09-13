"""CoSQ — Chain-of-Self-Questioning.

A three-stage self-interrogation framework that turns LLM answer generation into an
explicit decision: enumerate the sub-facts a question needs, judge certainty about
each, then answer from the certain ones or abstain.

Importing this package registers the built-in backends, strategies, decision rules,
and datasets, so :func:`cosq.registry.resolve` can find them by name.
"""

from cosq.backends import LLMBackend, MockBackend, load_backend
from cosq.backends import hf_local as _hf_local  # noqa: F401  (registers hf_local)
from cosq.backends import openai as _openai  # noqa: F401  (registers openai)
from cosq.config import ExperimentConfig, ModelConfig, config_hash
from cosq.data import MMLU, JsonlDataset, NaturalQuestionsShort, TruthfulQAMC1
from cosq.decision import DecisionRule, ThresholdRule
from cosq.strategies import (
    CoSQCriticalGroundedStrategy,
    CoSQGroundedAdaptiveStrategy,
    CoSQGroundedStrategy,
    CoSQStrategy,
    CoTAbstainStrategy,
    CoTStrategy,
    DirectStrategy,
    Strategy,
)
from cosq.types import (
    AnswerRecord,
    Certainty,
    Completion,
    Decision,
    GenerationParams,
    Outcome,
    Question,
    ScoredRecord,
)

__version__ = "0.2.0"

#: Convenient alias: ``CoSQ(backend=..., rule=...)``.
CoSQ = CoSQStrategy

__all__ = [
    "MMLU",
    "AnswerRecord",
    "Certainty",
    "CoSQ",
    "CoSQCriticalGroundedStrategy",
    "CoSQGroundedAdaptiveStrategy",
    "CoSQGroundedStrategy",
    "CoSQStrategy",
    "CoTAbstainStrategy",
    "CoTStrategy",
    "Completion",
    "Decision",
    "DecisionRule",
    "DirectStrategy",
    "ExperimentConfig",
    "GenerationParams",
    "JsonlDataset",
    "LLMBackend",
    "MockBackend",
    "ModelConfig",
    "NaturalQuestionsShort",
    "Outcome",
    "Question",
    "ScoredRecord",
    "Strategy",
    "ThresholdRule",
    "TruthfulQAMC1",
    "__version__",
    "config_hash",
    "load_backend",
]
