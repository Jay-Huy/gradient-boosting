from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    mode: str = "min"
    patience: int = 5
    min_delta: float = 0.0

    best_value: float | None = None
    best_step: int | None = None
    num_bad_epochs: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        self.mode = self.mode.lower().strip()
        if self.mode not in {"min", "max"}:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'")
        if self.patience <= 0:
            raise ValueError("EarlyStopping patience must be > 0")
        if self.min_delta < 0.0:
            raise ValueError("EarlyStopping min_delta must be >= 0")

    def reset(self) -> None:
        self.best_value = None
        self.best_step = None
        self.num_bad_epochs = 0
        self.stopped = False

    def state_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_value": self.best_value,
            "best_step": self.best_step,
            "num_bad_epochs": self.num_bad_epochs,
            "stopped": self.stopped,
        }

    def load_state_dict(self, state: dict[str, float | int | bool | None]) -> None:
        if "mode" in state:
            self.mode = str(state["mode"]).lower().strip()
        if "patience" in state:
            self.patience = int(state["patience"])
        if "min_delta" in state:
            self.min_delta = float(state["min_delta"])
        self.best_value = None if state.get("best_value") is None else float(state["best_value"])
        self.best_step = None if state.get("best_step") is None else int(state["best_step"])
        self.num_bad_epochs = int(state.get("num_bad_epochs", 0))
        self.stopped = bool(state.get("stopped", False))

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < (self.best_value - self.min_delta)
        return value > (self.best_value + self.min_delta)

    def update(self, value: float, step: int | None = None) -> bool:
        if self._is_improvement(float(value)):
            self.best_value = float(value)
            self.best_step = step
            self.num_bad_epochs = 0
            self.stopped = False
            return False

        self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.stopped = True
            return True
        return False
