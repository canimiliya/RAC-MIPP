"""Frozen benchmark and evaluation interfaces for RAC-MIPP."""

from .contracts import EvaluationRole, validate_evaluation_use
from .evaluator import evaluate_policy

__all__ = ["EvaluationRole", "evaluate_policy", "validate_evaluation_use"]
