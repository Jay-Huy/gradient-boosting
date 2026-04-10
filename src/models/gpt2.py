from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.base import BaseModel


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight
        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        _, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds block size {self.config.block_size}"
            )

        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss


class GPT2LanguageModel(BaseModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.gpt_config = GPTConfig(
            block_size=int(config["block_size"]),
            vocab_size=int(config["vocab_size"]),
            n_layer=int(config["n_layer"]),
            n_head=int(config["n_head"]),
            n_embd=int(config["n_embd"]),
            dropout=float(config.get("dropout", 0.0)),
            bias=bool(config.get("bias", True)),
        )
        self.backbone = GPT(self.gpt_config)

    def forward(self, batch: Any) -> dict[str, Any]:
        if not isinstance(batch, dict):
            raise TypeError("Expected batch to be a dict with keys input_ids and targets")

        input_ids = batch["input_ids"]
        targets = batch.get("targets")
        logits, loss = self.backbone(input_ids, targets=targets)

        if loss is None:
            loss = logits.sum() * 0.0

        predictions = torch.argmax(logits, dim=-1)
        return {
            "loss": loss,
            "metric_loss": loss,
            "predictions": predictions,
            "targets": targets,
            "logits": logits,
        }

    def infer(self, batch: Any) -> Any:
        if not isinstance(batch, dict):
            raise TypeError("Expected batch to be a dict with key input_ids")
        input_ids = batch["input_ids"]
        with torch.no_grad():
            logits, _ = self.backbone(input_ids, targets=None)
        return torch.argmax(logits, dim=-1)


class GPT2BoostingLanguageModel(BaseModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        self.num_learners = int(config["num_learners"])
        self.shrinkage = float(config.get("shrinkage", 1.0))
        weak_cfg = dict(config["weak_learner"])

        self.learners = nn.ModuleList([GPT(self._make_gpt_config(weak_cfg)) for _ in range(self.num_learners)])
        self.learner_alphas: list[float] = [0.0 for _ in range(self.num_learners)]
        self.active_learner_idx = 0
        self._stage_mse_loss = torch.nn.MSELoss()
        print(f"[model-init] weak_learner_params={sum(p.numel() for p in self.learners[0].parameters())}")
        self.set_active_learner(idx = 0)

    def _make_gpt_config(self, cfg: dict[str, Any]) -> GPTConfig:
        return GPTConfig(
            block_size=int(cfg["block_size"]),
            vocab_size=int(cfg["vocab_size"]),
            n_layer=int(cfg["n_layer"]),
            n_head=int(cfg["n_head"]),
            n_embd=int(cfg["n_embd"]),
            dropout=float(cfg.get("dropout", 0.0)),
            bias=bool(cfg.get("bias", True)),
        )

    def set_active_learner(self, idx: int) -> None:
        if idx < 0 or idx >= self.num_learners:
            raise ValueError(f"active learner index {idx} is out of range [0, {self.num_learners - 1}]")
        self.active_learner_idx = idx

        for learner_idx, learner in enumerate(self.learners):
            active_learner = learner_idx == idx
            for param in learner.parameters():
                param.requires_grad = active_learner

    def set_learner_alpha(self, idx: int, alpha: float) -> None:
        if idx < 0 or idx >= self.num_learners:
            raise ValueError(f"learner index {idx} is out of range [0, {self.num_learners - 1}]")
        self.learner_alphas[idx] = float(alpha)

    def begin_learner_stage(self, idx: int) -> None:
        self.set_active_learner(idx)
        self.set_learner_alpha(idx, 0.0)

    def get_learner_alpha(self, idx: int) -> float:
        if idx < 0 or idx >= self.num_learners:
            raise ValueError(f"learner index {idx} is out of range [0, {self.num_learners - 1}]")
        return float(self.learner_alphas[idx])

    def get_active_learner_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        active_learner = self.learners[self.active_learner_idx]
        logits, _ = active_learner(input_ids, targets=None)
        return logits

    def get_ensemble_logits(self, input_ids: torch.Tensor, upto_idx: int | None = None) -> torch.Tensor:
        if upto_idx is None:
            upto_idx = self.num_learners - 1
        if upto_idx < 0:
            vocab_size = self.learners[0].config.vocab_size
            return torch.zeros(
                input_ids.size(0),
                input_ids.size(1),
                vocab_size,
                dtype=torch.float32,
                device=input_ids.device,
            )

        logits_sum: torch.Tensor | None = None
        for learner_idx in range(upto_idx + 1):
            learner = self.learners[learner_idx]
            alpha = self.get_learner_alpha(learner_idx)
            if alpha == 0.0:
                continue

            learner.eval()
            with torch.no_grad():
                learner_logits, _ = learner(input_ids, targets=None)

            scaled_logits = learner_logits * self.shrinkage * alpha
            logits_sum = scaled_logits if logits_sum is None else logits_sum + scaled_logits

        if logits_sum is None:
            vocab_size = self.learners[0].config.vocab_size
            return torch.zeros(
                input_ids.size(0),
                input_ids.size(1),
                vocab_size,
                dtype=torch.float32,
                device=input_ids.device,
            )
        return logits_sum

    def _negative_ce_gradient(self, ensemble_logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(ensemble_logits, dim=-1)
        gradient = -probs
        
        # Take non-masked targets
        valid_mask = targets.ne(-1)
        safe_targets = targets.masked_fill(~valid_mask, 0)


        gradient.scatter_add_(
            dim=-1,
            index=safe_targets.unsqueeze(-1),
            src=valid_mask.unsqueeze(-1).to(gradient.dtype),
        )
        gradient = gradient * valid_mask.unsqueeze(-1).to(gradient.dtype)
        return gradient, valid_mask

    def _stage_loss(
        self,
        weak_logits: torch.Tensor,
        residual_target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).to(weak_logits.dtype)
        elementwise = self._stage_mse_loss(weak_logits, residual_target) * mask
        denom = (mask.sum() * weak_logits.size(-1)).clamp_min(1.0)
        return elementwise.sum() / denom

    def _mean_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        nll_sum = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction="sum",
        )
        token_count = targets.ne(-1).sum().clamp_min(1)
        return nll_sum / token_count

    def line_search_active_learner_alpha(
        self,
        dataloader: Any,
        *,
        alpha_bounds: tuple[float, float] = (0.0, 1.0),
        maxiter: int = 50,
    ) -> float:
        try:
            from scipy import optimize as opt
        except Exception as exc:  # pragma: no cover - explicit dependency guard
            raise RuntimeError("scipy is required for alpha line search") from exc

        original_alpha = self.get_learner_alpha(self.active_learner_idx)
        original_training_mode = self.training
        model_device = next(self.parameters()).device
        self.eval()

        cached_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(model_device)
                targets = batch["targets"].to(model_device)

                previous_logits = self.get_ensemble_logits(
                    input_ids,
                    upto_idx=self.active_learner_idx - 1,
                )
                active_logits = self.get_active_learner_logits(input_ids)
                token_count = int(targets.ne(-1).sum().item())
                if token_count <= 0:
                    continue
                cached_batches.append((previous_logits, active_logits, targets, token_count))

        if not cached_batches:
            self.set_learner_alpha(self.active_learner_idx, original_alpha)
            if original_training_mode:
                self.train()
            return original_alpha

        def objective(alpha_scalar: float) -> float:
            alpha_value = float(alpha_scalar)
            total_ce = 0.0
            total_tokens = 0
            for previous_logits, active_logits, targets, token_count in cached_batches:
                logits = previous_logits + (self.shrinkage * alpha_value * active_logits)
                ce_sum = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                    reduction="sum",
                )
                total_ce += float(ce_sum.item())
                total_tokens += token_count
            if total_tokens <= 0:
                return 0.0
            return total_ce / total_tokens

        result = opt.minimize_scalar(
            objective,
            bounds=alpha_bounds,
            method="bounded",
            options={"maxiter": int(maxiter)},
        )

        best_alpha = float(result.x)
        self.set_learner_alpha(self.active_learner_idx, best_alpha)
        if math.isnan(best_alpha):
            self.set_learner_alpha(self.active_learner_idx, original_alpha)
            if original_training_mode:
                self.train()
            return original_alpha
        if original_training_mode:
            self.train()
        return best_alpha

    def _aggregate_logits(self, input_ids: torch.Tensor, upto_idx: int) -> torch.Tensor:
        return self.get_ensemble_logits(input_ids, upto_idx=upto_idx)

    def forward(self, batch: Any, mode: str = "train") -> dict[str, Any]:
        if not isinstance(batch, dict):
            raise TypeError("Expected batch to be a dict with keys input_ids and targets")

        input_ids = batch["input_ids"]
        targets = batch.get("targets")

        previous_logits = self.get_ensemble_logits(input_ids, upto_idx=self.active_learner_idx - 1) # Để tính residual
        active_logits = self.get_active_learner_logits(input_ids) # Để train với residual
        residual_target, valid_mask = self._negative_ce_gradient(previous_logits, targets)
        stage_loss = self._stage_loss(active_logits, residual_target, valid_mask)

        if mode == "train":
            return {
                "loss": stage_loss,
                "learner_id": self.active_learner_idx + 1,
            }

        else:
            current_logits = previous_logits + self.shrinkage * self.get_learner_alpha(self.active_learner_idx) * active_logits
            predictions = torch.argmax(current_logits, dim=-1)

            if targets is None:
                metric_loss = current_logits.sum() * 0.0
            else:
                metric_loss = self._mean_cross_entropy(current_logits, targets)

            return {
                "loss": stage_loss,
                "metric_loss": metric_loss,
                "predictions": predictions,
                "targets": targets,
                "learner_id": self.active_learner_idx + 1,
            }

    def infer(self, batch: Any) -> Any:
        if not isinstance(batch, dict):
            raise TypeError("Expected batch to be a dict with key input_ids")

        input_ids = batch["input_ids"]
        with torch.no_grad():
            logits = self.get_ensemble_logits(input_ids, upto_idx=self.num_learners - 1)
        return torch.argmax(logits, dim=-1)
