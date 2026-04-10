from __future__ import annotations

from typing import Any

from src.data.base import BaseDataModule


class DefaultDataModule(BaseDataModule):
    def load_data(self) -> None:
        self.raw_data = {"train": [], "val": [], "test": []}

    def setup_dataloaders(self) -> None:
        self.train_dataloader = self.raw_data["train"]
        self.val_dataloader = self.raw_data["val"]
        self.test_dataloader = self.raw_data["test"]

    def format_output(self, predictions: Any) -> Any:
        return {"format": self.config.get("output_format", "json"), "predictions": predictions}
