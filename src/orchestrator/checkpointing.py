from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _model_state_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": 1,
        "model_state_dict": _model_state_dict(model),
        "metadata": metadata,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, checkpoint_path)


def load_checkpoint(
    model: Any,
    checkpoint_path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    metadata = checkpoint.get("metadata", {})
    learner_alphas = metadata.get("learner_alphas", [])
    if hasattr(model, "set_learner_alpha"):
        for learner_idx, alpha in enumerate(learner_alphas):
            model.set_learner_alpha(learner_idx, float(alpha))

    active_learner_idx = metadata.get("active_learner_idx")
    if active_learner_idx is not None and hasattr(model, "set_active_learner"):
        model.set_active_learner(int(active_learner_idx))

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
