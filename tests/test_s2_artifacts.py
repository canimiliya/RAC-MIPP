import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_s2_artifacts_and_gates():
    for name in ("benchmark_contract.json", "evaluator_parity_summary.json", "determinism_summary.json", "completion_gate_audit.json"):
        path = ROOT / "artifacts/s2/r0" / name
        assert path.is_file(), name
        json.loads(path.read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "artifacts/s2/r0/completion_gate_audit.json").read_text(encoding="utf-8"))
    assert all(audit["gates"].values())
    assert audit["long_training_started"] is False
    assert audit["s3_work_started"] is False
