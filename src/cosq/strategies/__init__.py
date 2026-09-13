"""Answering strategies — the manipulated variable of the experiment."""

from cosq.strategies.base import Strategy
from cosq.strategies.baselines import CoTAbstainStrategy, CoTStrategy, DirectStrategy
from cosq.strategies.cosq import CoSQStrategy
from cosq.strategies.cosq_critical_grounded import CoSQCriticalGroundedStrategy
from cosq.strategies.cosq_gate import CoSQGateStrategy
from cosq.strategies.cosq_graded import CoSQGradedStrategy
from cosq.strategies.cosq_graded_gate import CoSQGradedGateStrategy
from cosq.strategies.cosq_grounded import CoSQGroundedStrategy
from cosq.strategies.cosq_grounded_adaptive import CoSQGroundedAdaptiveStrategy

__all__ = [
    "CoSQCriticalGroundedStrategy",
    "CoSQGateStrategy",
    "CoSQGradedGateStrategy",
    "CoSQGradedStrategy",
    "CoSQGroundedAdaptiveStrategy",
    "CoSQGroundedStrategy",
    "CoSQStrategy",
    "CoTAbstainStrategy",
    "CoTStrategy",
    "DirectStrategy",
    "Strategy",
]
