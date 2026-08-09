"""Algorithm-neutral evaluation driver and artifact writer."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Protocol, Sequence

from .communication import NullCommunicationObserver, PassiveCommunicationObserver
from .contracts import EvaluationRole, validate_evaluation_use
from .schema import normalize_episode_metrics, validate_run_manifest


class PolicyAdapter(Protocol):
    algorithm: str

    def evaluate_seed(self, seed: int, communication_observer: Any) -> dict[str, Any]: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def evaluate_policy(
    adapter: PolicyAdapter,
    *,
    seeds: Sequence[int],
    role: EvaluationRole | str,
    purpose: str,
    output_dir: Path,
    run_metadata: dict[str, Any],
    acknowledge_final_test: bool = False,
    communication_hook: bool = False,
) -> dict[str, Any]:
    """Evaluate one policy with common metrics and auditable split semantics."""

    role = validate_evaluation_use(
        role, purpose, acknowledge_final_test=acknowledge_final_test
    )
    if not seeds or len(set(map(int, seeds))) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    rows = []
    communication = []
    for trial_id, seed in enumerate(seeds, start=1):
        observer = PassiveCommunicationObserver() if communication_hook else NullCommunicationObserver()
        raw = adapter.evaluate_seed(int(seed), observer)
        metrics = normalize_episode_metrics(raw)
        row = {"trial_id": trial_id, "seed": int(seed), **metrics}
        rows.append(row)
        communication.append(observer.snapshot())

    trials_path = output_dir / "trials.csv"
    with trials_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}
    for metric in ("entropy", "F1", "mission_return", "episode_length", "measurement_count", "path_length"):
        aggregate[metric] = _summary([float(row[metric]) for row in rows])
    summary = {
        "schema_version": 1,
        "algorithm": adapter.algorithm,
        "evaluation_role": role.value,
        "purpose": purpose,
        "test_use_acknowledged": bool(acknowledge_final_test),
        "seeds": list(map(int, seeds)),
        "started_at": started,
        "ended_at": _utc_now(),
        "metrics": aggregate,
        "future_metrics": {
            "communication_load": "NOT_AVAILABLE_YET",
            "packet_delivery": "NOT_AVAILABLE_YET",
            "tail_risk": "NOT_AVAILABLE_YET",
            "OOD_gap": "NOT_AVAILABLE_YET",
        },
        "communication_hook": {
            "enabled": communication_hook,
            "semantics": "PASSIVE_NO_ENVIRONMENT_MUTATION" if communication_hook else "HOOK_OFF",
            "episode_observations": communication,
        },
        "artifacts": [str(trials_path), str(output_dir / "summary.json"), str(output_dir / "run_manifest.json")],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        **run_metadata,
        "SEED": list(map(int, seeds)),
        "START_TIME": started,
        "END_TIME": summary["ended_at"],
        "STATUS": "COMPLETED",
        "PRIMARY_METRICS": aggregate,
        "ARTIFACT_PATHS": summary["artifacts"],
    }
    validate_run_manifest(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
