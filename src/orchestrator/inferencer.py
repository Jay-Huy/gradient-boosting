from __future__ import annotations

from typing import Any

from src.utils.factory import create_data_module, create_model


class Inferencer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.data_module = create_data_module(config["data"])
        self.model = create_model(config["model"])

    def run(self) -> None:
        self.data_module.load_data()
        self.data_module.setup_dataloaders()
        self.model.setup()
        self._infer()

    def _infer(self) -> None:
        raise NotImplementedError("Inference flow will be implemented later")