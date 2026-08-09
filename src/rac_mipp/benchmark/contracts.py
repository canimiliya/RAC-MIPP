"""Seed/split governance used by every RAC-MIPP algorithm."""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class EvaluationRole(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    IID_TEST = "IID_TEST"
    OOD_TEST = "OOD_TEST"

    @property
    def is_test(self) -> bool:
        return self in {self.IID_TEST, self.OOD_TEST}


_FORBIDDEN_TEST_PURPOSES = {
    "checkpoint_selection",
    "early_stopping",
    "hyperparameter_tuning",
    "model_development",
}


def validate_evaluation_use(
    role: EvaluationRole | str,
    purpose: str,
    *,
    acknowledge_final_test: bool = False,
) -> EvaluationRole:
    """Reject test leakage and require an explicit final-test acknowledgement."""

    role = EvaluationRole(role)
    normalized = purpose.strip().lower()
    if role.is_test and normalized in _FORBIDDEN_TEST_PURPOSES:
        raise ValueError(f"{role.value} cannot be used for {normalized}")
    if role.is_test and not acknowledge_final_test:
        raise ValueError(f"{role.value} requires acknowledge_final_test=True")
    return role


def validate_disjoint_seed_sets(seed_sets: dict[str, Iterable[int]]) -> None:
    seen: dict[int, str] = {}
    for role_name, values in seed_sets.items():
        for value in values:
            seed = int(value)
            if not 0 <= seed < 2**32:
                raise ValueError(f"seed outside uint32 range: {seed}")
            if seed in seen:
                raise ValueError(f"seed {seed} overlaps {seen[seed]} and {role_name}")
            seen[seed] = role_name
