from __future__ import annotations

from typing import Any

from src.orchestrator.inferencer import Inferencer
from src.orchestrator.trainer import Trainer


class Runner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self) -> None:
        mode = self.config.get("runtime", {}).get("mode", "train")
        if mode == "train":
            Trainer(self.config).run()
            return
        if mode == "infer":
            Inferencer(self.config).run()
            return
        raise ValueError(f"Unsupported mode: {mode}")
