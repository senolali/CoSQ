"""Answering strategies — the manipulated variable of the experiment."""

from cosq.strategies.base import Strategy
from cosq.strategies.baselines import CoTAbstainStrategy, CoTStrategy, DirectStrategy
from cosq.strategies.cosq import CoSQStrategy
from cosq.strategies.cosq_gate import CoSQGateStrategy
from cosq.strategies.cosq_graded import CoSQGradedStrategy
from cosq.strategies.cosq_graded_gate import CoSQGradedGateStrategy

__all__ = [
    "CoSQGateStrategy",
    "CoSQGradedGateStrategy",
    "CoSQGradedStrategy",
    "CoSQStrategy",
    "CoTAbstainStrategy",
    "CoTStrategy",
    "DirectStrategy",
    "Strategy",
]
