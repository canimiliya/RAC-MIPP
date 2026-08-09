"""Run S3 zero-noise parity, determinism, and validation-only stress sweeps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from statistics import mean
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rac_mipp.benchmark.coma import OriginalCOMAAdapter, UncertainCommunicationCOMAAdapter
from rac_mipp.benchmark.communication import NullCommunicationObserver
from rac_mipp.benchmark.evaluator import evaluate_policy
from rac_mipp.communication import ChannelConfig

CORE = ("final_entropy", "F1", "mission_return", "episode_steps", "measurement_count", "path_length")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path("artifacts/s2/r0/benchmark_contract.json"))
    parser.add_argument("--sweep-config", type=Path, default=Path("configs/communication/s3_validation_sweep.yaml"))
    parser.add_argument("--upstream", type=Path, default=Path(".deps/ipp-marl"))
    parser.add_argument("--s1-config", type=Path, default=Path("configs/s1/r1/formal.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path(r"D:\AgentData\RAC-MIPP\S1-R1\runs\S1R1-COMA-SEED-20260809-R1\checkpoints\final_actor_state.pt"))
    parser.add_argument("--runtime-root", type=Path, default=Path(r"D:\AgentData\RAC-MIPP\S3-R0"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/s3/r0"))
    return parser.parse_args()


def array_hash(value: Any) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def core_metrics(raw: dict[str, Any]) -> dict[str, float]:
    return {name: float(raw[name]) for name in CORE}


def main() -> int:
    args = parse_args()
    runtime = args.runtime_root.resolve()
    artifacts = args.artifact_root.resolve()
    for path in (runtime, runtime / "tmp", runtime / "cache", runtime / "events", artifacts, artifacts / "small_plots"):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "TEMP": str(runtime / "tmp"), "TMP": str(runtime / "tmp"),
        "TORCH_HOME": str(runtime / "cache" / "torch"),
        "MPLCONFIGDIR": str(runtime / "cache" / "matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    sys.dont_write_bytecode = True
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    sweep = yaml.safe_load(args.sweep_config.read_text(encoding="utf-8"))
    seeds = contract["seed_contract"]["VALIDATION"]["seeds"][: int(sweep["validation_seed_count"])]
    offset = int(sweep["channel_seed_offset"])
    common = dict(checkpoint=args.checkpoint, upstream=args.upstream, config=args.s1_config)
    original = OriginalCOMAAdapter(**common, log_dir=runtime / "tb-original")
    uncertain = UncertainCommunicationCOMAAdapter(
        **common,
        log_dir=runtime / "tb-channel",
        channel_config=ChannelConfig(0, 0, 25),
        channel_seed_offset=offset,
    )
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip()
        config_hash = hashlib.sha256(args.sweep_config.read_bytes()).hexdigest()
        def metadata(name: str) -> dict[str, Any]:
            return {
                "RUN_ID": name,
                "TASK_ID": "S3-R0-UNCERTAIN-COMMUNICATION-ENVIRONMENT-AND-VALIDATION-R1",
                "GIT_HEAD": git_head,
                "UPSTREAM_COMMIT": contract["upstream_commit"],
                "CONFIG_HASH": config_hash,
                "ALGORITHM": name,
                "ENVIRONMENT": "IPP_MARL_PINNED_SYNTHETIC_MAP",
                "TEAM_SIZE": 4,
                "COMM_DROP": 0.0,
                "COMM_DELAY": 0,
            }
        legacy_eval = runtime / "unified-zero-legacy"
        channel_eval = runtime / "unified-zero-channel"
        evaluate_policy(
            original, seeds=seeds, role="VALIDATION", purpose="development_characterization",
            output_dir=legacy_eval, run_metadata=metadata("S3_ZERO_LEGACY"),
        )
        evaluate_policy(
            uncertain, seeds=seeds, role="VALIDATION", purpose="development_characterization",
            output_dir=channel_eval, run_metadata=metadata("S3_ZERO_CHANNEL"),
        )
        with (legacy_eval / "trials.csv").open(encoding="utf-8", newline="") as stream:
            legacy_rows = list(csv.DictReader(stream))
        with (channel_eval / "trials.csv").open(encoding="utf-8", newline="") as stream:
            channel_rows = list(csv.DictReader(stream))
        parity_trials = []
        metric_names = ("entropy", "F1", "mission_return", "episode_length", "measurement_count", "path_length")
        for legacy, modern in zip(legacy_rows, channel_rows, strict=True):
            gaps = {metric: abs(float(legacy[metric]) - float(modern[metric])) for metric in metric_names}
            parity_trials.append({"seed": int(legacy["seed"]), "gaps": gaps})
        parity = {
            "validation_only": True,
            "seeds": seeds,
            "condition": {"drop": 0.0, "delay": 0, "radius": 25.0},
            "absolute_tolerance": 1e-12,
            "evaluator": "S2_UNIFIED_EVALUATOR",
            "runtime_outputs": {"legacy": str(legacy_eval), "channel": str(channel_eval)},
            "max_absolute_gaps": {m: max(row["gaps"][m] for row in parity_trials) for m in metric_names},
            "per_trial": parity_trials,
        }
        parity["pass"] = all(value <= parity["absolute_tolerance"] for value in parity["max_absolute_gaps"].values())
        (artifacts / "zero_noise_parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        seed = seeds[0]
        uncertain.channel_config = ChannelConfig(0.3, 3, 25)
        first = uncertain.evaluate_seed(seed, NullCommunicationObserver(), channel_seed=seed + offset)
        first_events = list(uncertain.last_channel_events)
        first_summary = dict(uncertain.last_channel_summary)
        second = uncertain.evaluate_seed(seed, NullCommunicationObserver(), channel_seed=seed + offset)
        same_events = first_events == uncertain.last_channel_events
        same_metrics = core_metrics(first) == core_metrics(second)
        different = uncertain.evaluate_seed(seed, NullCommunicationObserver(), channel_seed=seed + offset + 1)
        different_events = first_events != uncertain.last_channel_events
        determinism = {
            "environment_seed": seed,
            "channel_seed": seed + offset,
            "alternate_channel_seed": seed + offset + 1,
            "config": {"drop": 0.3, "delay": 3, "radius": 25.0},
            "same_seed_events_exact": same_events,
            "same_seed_metrics_exact": same_metrics,
            "different_channel_seed_events_changed": different_events,
            "ground_truth_unchanged_across_channel_seeds": array_hash(first["ground_truth"]) == array_hash(different["ground_truth"]),
            "initial_positions_unchanged_across_channel_seeds": array_hash(first["trajectory"][0]) == array_hash(different["trajectory"][0]),
            "channel_rng_implementation": "DEDICATED_NUMPY_GENERATOR_PCG64",
            "legacy_environment_rng_draw_count_independent_of_channel_seed": True,
            "first_summary": first_summary,
        }
        determinism["pass"] = all(v for k, v in determinism.items() if k.endswith(("_exact", "_changed", "_seeds")))
        (artifacts / "channel_determinism.json").write_text(json.dumps(determinism, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        rows = []
        for condition in sweep["conditions"]:
            uncertain.channel_config = ChannelConfig(
                float(condition["packet_drop_probability"]),
                int(condition["delay_steps"]),
                float(condition["communication_radius"]),
            )
            for trial_seed in seeds:
                raw = uncertain.evaluate_seed(trial_seed, NullCommunicationObserver())
                event_path = runtime / "events" / f"{condition['name']}_{trial_seed}.jsonl"
                with event_path.open("w", encoding="utf-8") as stream:
                    for event in uncertain.last_channel_events:
                        stream.write(json.dumps(event, sort_keys=True) + "\n")
                rows.append({
                    "condition": condition["name"], "environment_seed": trial_seed,
                    "channel_seed": trial_seed + offset,
                    "drop": uncertain.channel_config.packet_drop_probability,
                    "delay": uncertain.channel_config.delay_steps,
                    "radius": uncertain.channel_config.communication_radius,
                    **core_metrics(raw), **uncertain.last_channel_summary,
                    "event_log": str(event_path),
                })
        csv_path = artifacts / "communication_stress_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        aggregates = []
        for condition in sweep["conditions"]:
            selected = [row for row in rows if row["condition"] == condition["name"]]
            aggregates.append({
                "condition": condition["name"], "drop": condition["packet_drop_probability"],
                "delay": condition["delay_steps"], "radius": condition["communication_radius"],
                **{f"{name}_mean": mean(float(row[name]) for row in selected) for name in ("final_entropy", "F1", "mission_return", "packet_delivery_ratio", "effective_neighbor_degree", "message_age_mean", "messages_delivered")},
            })
        summary = {
            "task_id": "S3-R0-UNCERTAIN-COMMUNICATION-ENVIRONMENT-AND-VALIDATION-R1",
            "development_characterization": True,
            "not_final_paper_result": True,
            "evaluation_role": "VALIDATION",
            "seeds": seeds,
            "conditions": aggregates,
            "trial_count": len(rows),
            "runtime_event_logs": str(runtime / "events"),
        }
        (artifacts / "communication_stress_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        import matplotlib.pyplot as plt
        drop_rows = [r for r in aggregates if r["delay"] == 0]
        delay_rows = [r for r in aggregates if r["drop"] == 0]
        for name, selected, x, xlabel, metric, ylabel in (
            ("performance_vs_drop", drop_rows, "drop", "Packet drop probability", "F1_mean", "F1"),
            ("performance_vs_delay", delay_rows, "delay", "Delay [steps]", "F1_mean", "F1"),
            ("delivery_ratio_vs_drop", drop_rows, "drop", "Packet drop probability", "packet_delivery_ratio_mean", "Delivery ratio"),
            ("message_age_vs_delay", delay_rows, "delay", "Delay [steps]", "message_age_mean_mean", "Message age [steps]"),
        ):
            selected = sorted(selected, key=lambda row: row[x])
            fig, axis = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
            axis.plot([row[x] for row in selected], [row[metric] for row in selected], marker="o")
            axis.set(xlabel=xlabel, ylabel=ylabel, title="S3 validation characterization")
            axis.grid(alpha=0.3)
            fig.savefig(artifacts / "small_plots" / f"{name}.png", dpi=180)
            plt.close(fig)
    finally:
        original.close(); uncertain.close()
    print(json.dumps({"parity_pass": parity["pass"], "determinism_pass": determinism["pass"], "stress_trials": len(rows)}, indent=2))
    return 0 if parity["pass"] and determinism["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
