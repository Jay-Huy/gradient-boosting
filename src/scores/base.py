from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseScore(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def setup(self) -> None:
        return None

    def reset(self) -> None:
        return None

    @abstractmethod
    def update(
        self,
        predictions: Any,
        targets: Any,
        *,
        loss: float | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> dict[str, Any]:
        raise NotImplementedError
