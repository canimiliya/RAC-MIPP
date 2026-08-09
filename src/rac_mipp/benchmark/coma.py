"""Adapter for the pinned original ipp-marl COMA checkpoint."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml


EXPECTED_CHECKPOINT_SHA256 = "baf19c28c9140cecac18e0aa26baa8cd0b7bc1e11e37376e2b23f73438ef0e65"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OriginalCOMAAdapter:
    algorithm = "ORIGINAL_IPP_MARL_COMA"

    def __init__(self, *, checkpoint: Path, upstream: Path, config: Path, log_dir: Path):
        from scripts.s1 import run_formal_reproduction as s1

        self._s1 = s1
        checkpoint = checkpoint.resolve()
        if _sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("S1 COMA checkpoint SHA256 mismatch")
        framework = s1.configure_upstream(upstream)
        s1.install_adapters()
        params = yaml.safe_load((framework / "params.yaml").read_text(encoding="utf-8"))
        config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
        self.params = s1.deep_update(copy.deepcopy(params), config_data["overrides"])

        import torch
        from torch.utils.tensorboard import SummaryWriter
        from marl_framework.coma_wrapper import COMAWrapper
        from marl_framework.mapping.grid_maps import GridMap
        from marl_framework.sensors import Sensor
        from marl_framework.sensors.models import SensorModel

        self.writer = SummaryWriter(str(log_dir))
        self.wrapper = COMAWrapper(self.params, self.writer)
        self.wrapper.actor_network.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        self.wrapper.actor_network.eval()
        self.grid = GridMap(self.params)
        self.sensor = Sensor(SensorModel(), self.grid)

    def evaluate_seed(self, seed: int, communication_observer: Any) -> dict[str, Any]:
        return self._s1.evaluate_episode(
            self.params,
            self.wrapper,
            self.grid,
            self.sensor,
            seed,
            communication_observer=communication_observer,
        )

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
