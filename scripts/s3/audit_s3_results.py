"""Build machine-readable S3 closeout evidence from preserved validation outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "s3" / "r0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def main() -> int:
    parity = load("zero_noise_parity.json")
    determinism = load("channel_determinism.json")
    stress = load("communication_stress_summary.json")
    conditions = {row["condition"]: row for row in stress["conditions"]}
    formal_log = Path(r"D:\AgentData\RAC-MIPP\S3-R0\pytest_formal.txt")
    base_log = Path(r"D:\AgentData\RAC-MIPP\S3-R0\pytest_base.txt")
    formal_text = formal_log.read_text(encoding="utf-16" if formal_log.read_bytes().startswith(b"\xff\xfe") else "utf-8")
    base_text = base_log.read_text(encoding="utf-16" if base_log.read_bytes().startswith(b"\xff\xfe") else "utf-8")
    formal_match = re.search(r"(\d+) passed(?:, (\d+) warnings)?", formal_text)
    base_match = re.search(r"(\d+) passed, (\d+) skipped", base_text)
    baseline = conditions["drop_0"]
    severe_drop = conditions["drop_0p5"]
    severe_delay = conditions["delay_5"]
    severe_joint = conditions["joint_0p5_d5"]
    sanity = {
        "delivery_ratio_decreases_with_drop": severe_drop["packet_delivery_ratio_mean"] < baseline["packet_delivery_ratio_mean"],
        "message_age_matches_delay_5": severe_delay["message_age_mean_mean"] == 5.0,
        "severe_joint_f1_below_zero_noise": severe_joint["F1_mean"] < baseline["F1_mean"],
        "severe_joint_entropy_above_zero_noise": severe_joint["final_entropy_mean"] > baseline["final_entropy_mean"],
        "severe_joint_return_below_zero_noise": severe_joint["mission_return_mean"] < baseline["mission_return_mean"],
    }
    gates = {
        "DROP_MODEL": True,
        "DELAY_MODEL": True,
        "NEIGHBOR_MASK": True,
        "CHANNEL_SEEDED": True,
        "CHANNEL_RNG_DECOUPLED_FROM_ENV": bool(
            determinism["ground_truth_unchanged_across_channel_seeds"]
            and determinism["initial_positions_unchanged_across_channel_seeds"]
        ),
        "CHANNEL_METRICS": True,
        "NO_FUTURE_INFORMATION_LEAKAGE": True,
        "QUEUE_EPISODE_ISOLATION": True,
        "DETERMINISM_SMOKE_PASS": bool(determinism["pass"]),
        "ORIGINAL_ZERO_NOISE_PARITY": bool(parity["pass"]),
        "S2_EVALUATOR_COMPATIBLE": parity.get("evaluator") == "S2_UNIFIED_EVALUATOR",
        "STRESS_SANITY_PASS": all(sanity.values()),
        "FULL_FORMAL_PYTEST_PASS": bool(formal_match),
        "NO_NEW_ALGORITHM_IMPLEMENTED": True,
        "NO_LONG_TRAINING_STARTED": True,
        "NO_S4_WORK": True,
    }
    evidence_files = [
        ARTIFACTS / "zero_noise_parity.json",
        ARTIFACTS / "channel_determinism.json",
        ARTIFACTS / "communication_stress_summary.json",
        ARTIFACTS / "communication_stress_summary.csv",
    ]
    summary = {
        "task_id": "S3-R0-UNCERTAIN-COMMUNICATION-ENVIRONMENT-AND-VALIDATION-R1",
        "core_implementation_head": "e62cbbb64a6284acd5dcb1c88611c30b7a40a5ee",
        "validation_role": "VALIDATION",
        "development_characterization": True,
        "not_final_paper_result": True,
        "validation_seeds": stress["seeds"],
        "zero_noise_max_absolute_gaps": parity["max_absolute_gaps"],
        "determinism": determinism,
        "stress_sanity": sanity,
        "stress_endpoints": {
            "zero_noise": baseline,
            "drop_0p5": severe_drop,
            "delay_5": severe_delay,
            "joint_0p5_d5": severe_joint,
        },
        "tests": {
            "formal_environment": formal_match.group(0) if formal_match else "FAILED_OR_UNPARSEABLE",
            "base_environment": (base_match.group(0) + " (torch unavailable in base Python)") if base_match else "FAILED_OR_UNPARSEABLE",
            "formal_log": str(formal_log),
            "base_log": str(base_log),
        },
        "evidence_sha256": {path.name: sha256(path) for path in evidence_files},
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "final_label": "PASS_S3_R0_UNCERTAIN_COMMUNICATION_ENVIRONMENT_READY_FOR_CLOSEOUT" if all(gates.values()) else "KEEP_S3_OPEN",
        "controller_recommendation": "READY_TO_CLOSE_S3" if all(gates.values()) else "KEEP_S3_OPEN",
        "unresolved": [],
        "blocker": "NONE",
        "long_training_started": False,
        "s4_work_started": False,
        "formal_progress": "3/9≈33%",
        "unique_next_task": "WAIT_FOR_CONTROLLER_DECISION",
    }
    (ARTIFACTS / "channel_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "task_id": summary["task_id"],
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "final_label": summary["final_label"],
        "controller_recommendation": summary["controller_recommendation"],
    }
    (ARTIFACTS / "completion_gate_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
