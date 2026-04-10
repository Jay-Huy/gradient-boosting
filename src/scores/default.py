from __future__ import annotations

from typing import Any

from src.scores.base import BaseScore


class DefaultScore(BaseScore):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._items: list[tuple[Any, Any]] = []

    def update(
        self,
        predictions: Any,
        targets: Any,
        *,
        loss: float | None = None,
    ) -> None:
        self._items.append((predictions, targets))

    def reset(self) -> None:
        self._items.clear()

    def compute(self) -> dict[str, Any]:
        return {metric: 0.0 for metric in self.config.get("metrics", ["loss"])}
