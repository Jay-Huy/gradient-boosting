from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDataModule(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.raw_data = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    @abstractmethod
    def load_data(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def setup_dataloaders(self) -> None:
        raise NotImplementedError

    def format_output(self, predictions: Any) -> Any:
        return predictions
