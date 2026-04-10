from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

    def setup(self) -> None:
        return None

    @abstractmethod
    def forward(self, batch: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def infer(self, batch: Any) -> Any:
        raise NotImplementedError
