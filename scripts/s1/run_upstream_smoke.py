"""Run a bounded ipp-marl environment, train, checkpoint, and eval smoke.

The upstream checkout stays unmodified. This adapter only redirects runtime
paths and repairs the upstream BatchMemory mask accessor typo in-process.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import yaml


TASK_ID = "S1-R0-UPSTREAM-REPRO-CONTRACT-ENV-SMOKE-R1"
UPSTREAM_COMMIT = "1e9bdc3ba90f707ce79797468f533f5733c65e4b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path(".deps/ipp-marl"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/s1/r0/smoke.yaml")
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(r"D:\AgentData\RAC-MIPP\S1-R0"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/s1/r0/repro_smoke_summary.json"),
    )
    return parser.parse_args()


def deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_runtime(runtime_root: Path) -> dict[str, Path]:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    paths = {
        "tmp": runtime_root / "tmp",
        "runs": runtime_root / "runs" / "upstream_smoke",
        "torch_cache": runtime_root / "cache" / "torch",
        "matplotlib_cache": runtime_root / "cache" / "matplotlib",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(paths["tmp"])
    os.environ["TMP"] = str(paths["tmp"])
    os.environ["TORCH_HOME"] = str(paths["torch_cache"])
    os.environ["MPLCONFIGDIR"] = str(paths["matplotlib_cache"])
    return paths


def configure_upstream_imports(upstream: Path) -> Path:
    upstream = upstream.resolve()
    framework = upstream / "marl_framework"
    if not framework.is_dir():
        raise FileNotFoundError(f"Missing upstream framework: {framework}")
    for path in (str(framework), str(upstream)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return framework


def patch_mask_accessor() -> str:
    from marl_framework.batch_memory import BatchMemory

    original_get = BatchMemory.get

    def compatible_get(self, t: int, agent_id: int, argument: str):
        if argument == "mask":
            return self.transitions[agent_id][t].mask
        return original_get(self, t, agent_id, argument)

    BatchMemory.get = compatible_get
    return "BatchMemory.get('mask'): transition.masks -> transition.mask"


def tensor_finite(value) -> bool:
    import torch

    return bool(torch.isfinite(value.detach()).all().item())


def main() -> int:
    args = parse_args()
    started = time.time()
    runtime_paths = configure_runtime(args.runtime_root.resolve())
    framework = configure_upstream_imports(args.upstream)

    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from marl_framework.batch_memory import BatchMemory
    from marl_framework.coma_wrapper import COMAWrapper
    from marl_framework.mapping.grid_maps import GridMap
    from marl_framework.missions.episode_generator import EpisodeGenerator
    from marl_framework.sensors import Sensor
    from marl_framework.sensors.models import SensorModel

    compatibility_change = patch_mask_accessor()

    smoke_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    upstream_config_path = framework / "params.yaml"
    params = yaml.safe_load(upstream_config_path.read_text(encoding="utf-8"))
    params = deep_update(copy.deepcopy(params), smoke_config["overrides"])

    seed = int(smoke_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cuda = {
        "available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        device = torch.device("cuda")
        probe = torch.randn(256, 256, device=device, requires_grad=True)
        probe_loss = (probe @ probe.T).square().mean()
        probe_loss.backward()
        torch.cuda.synchronize()
        cuda.update(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "kernel_backward_pass": tensor_finite(probe.grad),
                "probe_loss": float(probe_loss.detach().cpu()),
            }
        )

    writer = SummaryWriter(str(runtime_paths["runs"] / "tensorboard"))
    checkpoint_path = runtime_paths["runs"] / "smoke_actor_state.pt"
    try:
        grid = GridMap(params)
        sensor = Sensor(SensorModel(), grid)
        wrapper = COMAWrapper(params, writer)
        memory = BatchMemory(params, wrapper)
        episode = EpisodeGenerator(params, writer, grid, sensor)

        train_result = episode.execute(
            int(smoke_config["episode_index_train"]), memory, wrapper, "train"
        )
        transition_count = memory.size()
        memory.build_td_targets(wrapper.target_critic_network)
        batches = memory.build_batches()
        critic_before = [p.detach().clone() for p in wrapper.critic_network.parameters()]
        actor_before = [p.detach().clone() for p in wrapper.actor_network.parameters()]
        q_values, critic_metrics = wrapper.critic_learner.learn(0, batches, 0)
        actor_network, actor_metrics = wrapper.actor_learner.learn(
            batches, q_values, train_result[7]
        )
        critic_changed = any(
            not torch.equal(before, after.detach())
            for before, after in zip(critic_before, wrapper.critic_network.parameters())
        )
        actor_changed = any(
            not torch.equal(before, after.detach())
            for before, after in zip(actor_before, actor_network.parameters())
        )

        torch.save(actor_network.state_dict(), checkpoint_path)
        checkpoint_hash = sha256(checkpoint_path)

        eval_wrapper = COMAWrapper(params, writer)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        eval_wrapper.actor_network.load_state_dict(loaded)
        eval_memory = BatchMemory(params, eval_wrapper)
        eval_episode = EpisodeGenerator(params, writer, GridMap(params), sensor)
        eval_result = eval_episode.execute(
            int(smoke_config["episode_index_eval"]),
            eval_memory,
            eval_wrapper,
            "eval",
        )
    finally:
        writer.flush()
        writer.close()

    summary = {
        "task_id": TASK_ID,
        "smoke_only": True,
        "not_paper_result": True,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_unmodified": True,
        "config": {
            "path": str(args.config.resolve()),
            "sha256": sha256(args.config),
            "overrides": smoke_config["overrides"],
        },
        "runtime_root": str(args.runtime_root.resolve()),
        "compatibility_changes": [compatibility_change],
        "cuda": cuda,
        "gates": {
            "upstream_import": True,
            "environment_construction": True,
            "environment_reset": True,
            "environment_step": transition_count > 0,
            "multi_agent_observation": transition_count
            == params["experiment"]["missions"]["n_agents"]
            * (params["experiment"]["constraints"]["budget"] + 1),
            "action_pipeline": len(train_result[8]) > 0,
            "episode_termination": len(train_result[1])
            == params["experiment"]["constraints"]["budget"] + 1,
            "reward_logging": bool(all(np.isfinite(train_result[1]))),
            "model_forward": len(q_values) > 0,
            "loss_computation": tensor_finite(critic_metrics[0])
            and tensor_finite(actor_metrics[0]),
            "backward": all(
                parameter.grad is None or tensor_finite(parameter.grad)
                for parameter in list(wrapper.actor_network.parameters())
                + list(wrapper.critic_network.parameters())
            ),
            "optimizer_step": actor_changed and critic_changed,
            "checkpoint_write": checkpoint_path.is_file(),
            "checkpoint_load": True,
            "evaluation_episode": len(eval_result[1])
            == params["experiment"]["constraints"]["budget"] + 1,
            "metric_output": bool(np.isfinite(eval_result[0])),
            "qualitative_output": len(eval_result[5]) > 0,
        },
        "train_smoke": {
            "episode_return": float(train_result[0]),
            "absolute_return": float(train_result[2]),
            "rewards": [float(value) for value in train_result[1]],
            "transitions": transition_count,
            "batches": len(batches),
            "critic_loss": float(critic_metrics[0].detach().cpu()),
            "actor_loss": float(actor_metrics[0].detach().cpu()),
            "actor_parameters_changed": actor_changed,
            "critic_parameters_changed": critic_changed,
        },
        "eval_smoke": {
            "episode_return": float(eval_result[0]),
            "absolute_return": float(eval_result[2]),
            "rewards": [float(value) for value in eval_result[1]],
            "trajectory_steps": len(eval_result[5]),
            "trajectory_positions_m": [
                [position.tolist() for position in timestep]
                for timestep in eval_result[5]
            ],
            "ground_truth_shape": list(eval_result[3].shape),
            "ground_truth_positive_fraction": float(np.mean(eval_result[3])),
        },
        "local_only_artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "tensorboard": str(runtime_paths["runs"] / "tensorboard"),
        },
        "long_training_started": False,
        "formal_repro_training_started": False,
        "attempt_history": [
            {
                "attempt": "initial_evidence_write",
                "status": "FAILED_AFTER_PIPELINE",
                "reason": "NumPy bool_ was not JSON serializable",
                "scientific_pipeline_completed": True,
                "resolution": "Cast machine-readable gate values to built-in bool",
            }
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary["all_smoke_gates_pass"] = all(summary["gates"].values()) and bool(
        summary["cuda"].get("kernel_backward_pass")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_smoke_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
