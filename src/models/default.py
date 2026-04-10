from __future__ import annotations

from typing import Any

import torch

from src.models.base import BaseModel


class DefaultModel(BaseModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch: Any) -> dict[str, Any]:
        # Keep a connected graph so optimizer step works even with placeholder data.
        loss = self._anchor.pow(2).mean()
        predictions = batch
        return {"loss": loss, "predictions": predictions}

    def infer(self, batch: Any) -> Any:
        with torch.no_grad():
            output = self.forward(batch)
        return output.get("predictions")
