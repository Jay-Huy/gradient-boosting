from __future__ import annotations

import math
from typing import Any

import torch

from src.scores.base import BaseScore


class PerplexityScore(BaseScore):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._nll_sum = 0.0
        self._token_count = 0

    def reset(self) -> None:
        self._nll_sum = 0.0
        self._token_count = 0

    def update(
        self,
        predictions: Any,
        targets: Any,
        *,
        loss: float | None = None,
    ) -> None:
        if loss is None:
            return

        token_count = 1
        if isinstance(targets, torch.Tensor):
            valid_mask = targets != -1
            token_count = int(valid_mask.sum().item())
            if token_count <= 0:
                return

        self._nll_sum += float(loss) * float(token_count)
        self._token_count += token_count

    def compute(self) -> dict[str, Any]:
        if self._token_count == 0:
            return {"perplexity": 0.0, "loss": 0.0}
        avg_loss = self._nll_sum / self._token_count
        return {
            "perplexity": float(math.exp(avg_loss)),
            "loss": float(avg_loss),
        }
