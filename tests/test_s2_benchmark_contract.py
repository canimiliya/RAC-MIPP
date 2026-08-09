import json
from pathlib import Path

import pytest

from rac_mipp.benchmark.communication import NullCommunicationObserver, PassiveCommunicationObserver
from rac_mipp.benchmark.contracts import EvaluationRole, validate_disjoint_seed_sets, validate_evaluation_use
from rac_mipp.benchmark.evaluator import evaluate_policy
from rac_mipp.benchmark.schema import REQUIRED_RUN_FIELDS, validate_run_manifest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "artifacts/s2/r0/benchmark_contract.json").read_text(encoding="utf-8"))


def test_seed_sets_are_frozen_disjoint_and_preserve_s1_anchor():
    sets = {name: data["seeds"] for name, data in CONTRACT["seed_contract"].items() if isinstance(data, dict)}
    validate_disjoint_seed_sets(sets)
    assert sets["IID_TEST"] == list(range(20001, 20051))
    assert len(sets["TRAIN"]) == 10


def test_test_roles_reject_leakage_and_require_acknowledgement():
    with pytest.raises(ValueError, match="cannot be used"):
        validate_evaluation_use("IID_TEST", "checkpoint_selection", acknowledge_final_test=True)
    with pytest.raises(ValueError, match="acknowledge"):
        validate_evaluation_use("OOD_TEST", "final_reporting")
    assert validate_evaluation_use("VALIDATION", "checkpoint_selection") is EvaluationRole.VALIDATION


def test_run_schema_validator_refuses_missing_seed():
    payload = {field: "x" for field in REQUIRED_RUN_FIELDS}
    payload.update({"STATUS": "COMPLETED", "PRIMARY_METRICS": {}, "ARTIFACT_PATHS": []})
    payload.pop("SEED")
    with pytest.raises(ValueError, match="SEED"):
        validate_run_manifest(payload)


class _FakeAdapter:
    algorithm = "FAKE"

    def evaluate_seed(self, seed, communication_observer):
        communication_observer.observe_positions([(0, 0), (1, 0)], 2.0)
        return {"final_entropy": seed / 100, "F1": 0.5, "mission_return": 1, "episode_steps": 3, "measurement_count": 6, "path_length": 2}


def test_common_evaluator_is_deterministic_and_marks_future_metrics(tmp_path):
    metadata = {
        "RUN_ID": "fake",
        "TASK_ID": "test",
        "GIT_HEAD": "a" * 40,
        "UPSTREAM_COMMIT": "b" * 40,
        "CONFIG_HASH": "c" * 64,
        "ALGORITHM": "FAKE",
        "ENVIRONMENT": "FAKE",
        "TEAM_SIZE": 2,
        "COMM_DROP": "NOT_AVAILABLE_YET",
        "COMM_DELAY": "NOT_AVAILABLE_YET",
    }
    first = evaluate_policy(_FakeAdapter(), seeds=[1, 2], role="VALIDATION", purpose="development", output_dir=tmp_path / "a", run_metadata=metadata, communication_hook=True)
    second = evaluate_policy(_FakeAdapter(), seeds=[1, 2], role="VALIDATION", purpose="development", output_dir=tmp_path / "b", run_metadata=metadata, communication_hook=True)
    assert first["metrics"] == second["metrics"]
    assert first["future_metrics"]["tail_risk"] == "NOT_AVAILABLE_YET"
    assert first["communication_hook"]["episode_observations"][0]["proximity_directed_links"] == 2
    manifest = json.loads((tmp_path / "a/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["SEED"] == [1, 2]


def test_null_and_passive_hooks_return_no_environment_controls():
    null = NullCommunicationObserver()
    passive = PassiveCommunicationObserver()
    assert null.observe_positions([(0, 0)], 1.0) is None
    assert passive.observe_positions([(0, 0)], 1.0) is None
    assert null.snapshot()["messages_attempted"] == "NOT_AVAILABLE_YET"
