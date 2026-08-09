import csv
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/s1/r1"


def test_table1_trials_are_complete_and_recompute_summary():
    with (ARTIFACTS / "table1_trials.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 50
    assert [int(row["trial_id"]) for row in rows] == list(range(1, 51))
    assert [int(row["seed"]) for row in rows] == list(range(20001, 20051))
    required = ("final_entropy", "F1", "mission_return", "absolute_return")
    assert all(math.isfinite(float(row[field])) for row in rows for field in required)

    summary = json.loads((ARTIFACTS / "table1_summary.json").read_text(encoding="utf-8"))
    entropies = [float(row["final_entropy"]) for row in rows]
    f1_values = [float(row["F1"]) for row in rows]
    assert math.isclose(statistics.mean(entropies), summary["entropy_mean"], abs_tol=1e-12)
    assert math.isclose(statistics.stdev(entropies), summary["entropy_std"], abs_tol=1e-12)
    assert math.isclose(statistics.mean(f1_values), summary["F1_mean"], abs_tol=1e-12)
    assert math.isclose(statistics.stdev(f1_values), summary["F1_std"], abs_tol=1e-12)


def test_strong_parity_and_leakage_barrier():
    summary = json.loads((ARTIFACTS / "table1_summary.json").read_text(encoding="utf-8"))
    assert summary["n"] == 50
    assert summary["entropy_standardized_gap"] <= 1.0
    assert summary["F1_standardized_gap"] <= 1.0
    assert summary["quantitative_parity"] == "STRONG_PARITY"
    assert summary["checkpoint_selection"] == "FINAL_CHECKPOINT_ONLY"
    assert summary["test_leakage"] is False
    assert (ARTIFACTS / "small_plots/formal_qualitative_rollout.png").stat().st_size > 0


def test_tracked_run_manifest_and_completion_gates():
    manifest = json.loads(
        (ARTIFACTS / "formal_run_manifest.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (ARTIFACTS / "completion_gate_audit.json").read_text(encoding="utf-8")
    )
    assert manifest["formal_run"] is True
    assert manifest["status"] == "COMPLETED"
    assert manifest["completed_updates"] == 1500
    assert manifest["agent_transitions"] == 4_500_000
    assert manifest["missions"] == 75_000
    assert all(audit["gates"].values())
    assert audit["quantitative_parity"] == "STRONG_PARITY"
    assert audit["test_leakage"] is False
