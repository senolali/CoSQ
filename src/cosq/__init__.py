"""CoSQ - Chain-of-Self-Questioning.

CoSQ is a lightweight framework for selective factual answering with language
models. It asks a model to decompose a question into required knowledge items,
estimate whether those items are supported, and then either answer or abstain.
"""

from cosq.backends import LLMBackend, MockBackend, load_backend
from cosq.backends import hf_local as _hf_local  # noqa: F401
from cosq.backends import openai as _openai  # noqa: F401
from cosq.config import ExperimentConfig, ModelConfig, config_hash
from cosq.data import JsonlDataset, TruthfulQAMC1
from cosq.decision import DecisionRule, ThresholdRule
from cosq.strategies import (
    CoSQGateStrategy,
    CoSQGradedGateStrategy,
    CoSQGradedStrategy,
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

__version__ = "0.1.0"

CoSQ = CoSQStrategy

__all__ = [
    "AnswerRecord", "Certainty", "CoSQ", "CoSQGateStrategy", "CoSQGradedGateStrategy",
    "CoSQGradedStrategy", "CoSQStrategy", "CoTAbstainStrategy", "CoTStrategy",
    "Completion", "Decision", "DecisionRule", "DirectStrategy", "ExperimentConfig",
    "GenerationParams", "JsonlDataset", "LLMBackend", "MockBackend", "ModelConfig",
    "Outcome", "Question", "ScoredRecord", "Strategy", "ThresholdRule", "TruthfulQAMC1",
    "__version__", "config_hash", "load_backend",
]
