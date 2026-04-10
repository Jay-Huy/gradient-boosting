from __future__ import annotations

from collections.abc import Iterable
import json
import math
from logging import Logger
from pathlib import Path
from statistics import mean
from time import perf_counter
from datetime import datetime, timezone
from typing import Any
import tqdm
import torch

from src.utils.factory import create_data_module, create_model, create_score
from src.utils.logging import get_logger


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        # Initialize components
        self.data_module = create_data_module(config["data"])
        self.model = create_model(config["model"])
        self.score = create_score(config.get("score", {}))

    def run(self) -> None:
        run_start = datetime.now(timezone.utc)

        # Load data and dataloaders
        self.data_module.load_data()
        self.data_module.setup_dataloaders()
        train_dataloader = self.data_module.train_dataloader
        val_dataloader = self.data_module.val_dataloader

        # Set up training config and device
        self.device = self._resolve_device(str(self.config["training"].get("device", "auto")))

        # Set up model
        self.model.setup() # None at the moment
        self.model.to(self.device)

        # Set up score
        self.score.setup() # None at the moment

        # Do not allow default training config
        training_mode = str(self.config["training"].get("mode", "baseline")).lower()
        epochs = int(self.config["training"]["epochs"])
        eval_step = int(self.config["training"]["eval_step"])

        logger = get_logger("trainer", self.config["training"].get("log_file", "outputs/logs/train.log"))
        run_id = self._build_run_id()

        results: dict[str, Any] = {
            "metadata": {
                "run_id": run_id,
                "started_at": run_start.isoformat(),
                "finished_at": None,
                "config": self.config,
                "data_meta": self._collect_data_meta(),
                "score_meta": {
                    "metrics": self.config.get("score", {}).get("params", {}).get("metrics", ["loss"]),
                    "primary_metric": self._primary_score_name(),
                },
                "training_mode": training_mode,
            },
            "train": {
                "metadata": {"split": "train", "num_samples": 0},
                "losses": [],
                "scores": {},
                "records": [],
            },
            "val": {
                "metadata": {"split": "val", "num_samples": 0},
                "losses": [],
                "scores": {},
                "records": [],
            },
            "summary": {},
        }
        
        if training_mode == "boosting":
            self._run_boosting(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=epochs,
                eval_step=eval_step,
                logger=logger,
                results=results,
            )
        else:
            optimizer = self._build_optimizer(self.config["training"]['optimizer'])
            self._train_loop(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=epochs,
                eval_step=eval_step,
                optimizer=optimizer,
                logger=logger,
                results=results,
                training_mode=training_mode,
                learner_id=None,
            )

        run_end = datetime.now(timezone.utc)
        results["metadata"]["finished_at"] = run_end.isoformat()
        results["summary"] = self._build_summary(results)

        output_path = self._resolve_results_path(run_id)
        self._write_results_json(output_path, results)
        logger.info("results_json=%s", output_path.as_posix())

    def _run_boosting(
        self,
        *,
        train_dataloader: Iterable[Any] | None,
        val_dataloader: Iterable[Any] | None,
        epochs: int,
        eval_step: int,
        logger: Logger,
        results: dict[str, Any],
    ) -> None:
        
        boosting_config = self.config["training"]['boosting']

        # Line search Config
        alpha_split = str(boosting_config['line_search']['split']).lower()
        alpha_bounds = tuple(boosting_config['line_search']['alpha_bounds'])

        num_learners = self.model.num_learners

        results["metadata"]["boosting"] = {
            "num_learners": num_learners,
            "epochs_per_learner": self.config["training"]['epochs'],
            "shrinkage": self.config["model"]['params']['shrinkage'],
            "line_search": {
                "split": alpha_split,
                "alpha_bounds": list(alpha_bounds),
            },
            "learner_alphas": [],
        }

        for learner_idx in range(num_learners):
            # Set active learner in the model for current stage
            self.model.begin_learner_stage(learner_idx)

            # Init specific learner optimizer
            optimizer = self._build_optimizer(self.config["training"]['optimizer'])
            logger.info(
                "mode=boosting stage_start learner=%d/%d epochs=%d",
                learner_idx + 1,
                num_learners,
                epochs,
            )

            self._train_loop(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=epochs,
                eval_step=eval_step,
                optimizer=optimizer,
                logger=logger,
                results=results,
                training_mode="boosting",
                learner_id=learner_idx + 1,
            )

            search_loader = val_dataloader if alpha_split == "val" and val_dataloader is not None else train_dataloader
            alpha = 0.0
            if search_loader is not None and hasattr(self.model, "line_search_active_learner_alpha"):
                logger.info(
                    "mode=boosting phase=alpha-search learner=%d split=%s alpha_bounds=%s status=start",
                    learner_idx + 1,
                    alpha_split,
                    list(alpha_bounds),
                )
                alpha = float(
                    self.model.line_search_active_learner_alpha(
                        search_loader,
                        alpha_bounds=alpha_bounds,
                    )
                )
            if hasattr(self.model, "set_learner_alpha"):
                self.model.set_learner_alpha(learner_idx, alpha)

            results["metadata"]["boosting"]["learner_alphas"].append(float(alpha))
            logger.info(
                "mode=boosting phase=alpha-search learner=%d split=%s status=done best_alpha=%.6f",
                learner_idx + 1,
                alpha_split,
                alpha,
            )

            if val_dataloader is not None:
                eval_start = perf_counter()
                eval_stats = self._run_dataloader(
                    val_dataloader,
                    is_train=False,
                    optimizer=None,
                    score_mode="default",
                    split_name="val-post-alpha",
                )
                eval_duration = perf_counter() - eval_start
                eval_record = {
                    "epoch": None,
                    "split": "val",
                    "training_mode": "boosting",
                    "learner_id": learner_idx + 1,
                    "post_line_search": True,
                    "num_samples": int(eval_stats["num_samples"]),
                    "num_batches": int(eval_stats["num_batches"]),
                    "avg_loss": eval_stats["loss"],
                    "avg_score": eval_stats["score"],
                    "score_name": eval_stats["score_name"],
                    "duration_sec": eval_duration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._append_split_record(results, "val", eval_record)
                logger.info(
                    "mode=boosting phase=val-post-alpha learner=%d avg_loss=%.6f avg_%s=%.6f samples=%d batches=%d time=%.2fs",
                    learner_idx + 1,
                    eval_stats["loss"],
                    eval_stats["score_name"],
                    eval_stats["score"],
                    int(eval_stats["num_samples"]),
                    int(eval_stats["num_batches"]),
                    eval_duration,
                )

            logger.info("mode=boosting stage_end learner=%d/%d", learner_idx + 1, num_learners)

    def _build_optimizer(self, optimizer_config: dict[str, Any]) -> torch.optim.Optimizer:
        name = str(optimizer_config.get("name", "adamw")).lower()
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("Model has no trainable parameters for optimizer initialization")

        if name == "adamw":
            lr = float(optimizer_config.get("lr", 1e-3))
            return torch.optim.AdamW(
                params,
                lr=lr,
                betas=tuple(optimizer_config.get("betas", (0.9, 0.999))),
                eps=float(optimizer_config.get("eps", 1e-8)),
                weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
            )
        if name == "adam":
            lr = float(optimizer_config.get("lr", 1e-3))
            return torch.optim.Adam(
                params,
                lr=lr,
                betas=tuple(optimizer_config.get("betas", (0.9, 0.999))),
                eps=float(optimizer_config.get("eps", 1e-8)),
                weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
            )
        if name == "sgd":
            lr = float(optimizer_config.get("lr", 1e-2))
            return torch.optim.SGD(
                params,
                lr=lr,
                momentum=float(optimizer_config.get("momentum", 0.0)),
                weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
                nesterov=bool(optimizer_config.get("nesterov", False)),
            )
        raise ValueError(f"Unsupported optimizer: {name}")

    def _process_batch(
        self,
        batch: Any,
        *,
        is_train: bool,
        optimizer: torch.optim.Optimizer | None,
        score_mode: str = "default",
    ) -> dict[str, float]:
        batch = self._move_batch_to_device(batch)

        if is_train:
            if optimizer is None:
                raise ValueError("Optimizer is required when is_train=True")
            self.model.train()
            optimizer.zero_grad()
            output = self.model.forward(batch)
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model.forward(batch)

        if not isinstance(output, dict):
            raise TypeError("Model forward output must be a dict and include a 'loss' key")

        loss = output.get("loss")
        if loss is None:
            raise KeyError("Model forward output is missing required key: 'loss'")

        if not isinstance(loss, torch.Tensor):
            loss = torch.as_tensor(loss, dtype=torch.float32)

        if is_train:
            loss.backward()
            optimizer.step()

        if score_mode == "pre_alpha":
            predictions = output.get("pre_alpha_predictions", output.get("predictions"))
            metric_loss = output.get("pre_alpha_metric_loss", output.get("metric_loss", loss))
        else:
            predictions = output.get("predictions")
            metric_loss = output.get("metric_loss", loss)

        targets = output.get("targets")
        if targets is None and isinstance(batch, dict):
            targets = batch.get("targets")
            if targets is None:
                targets = batch.get("labels")

        score_loss = float(metric_loss.detach().item() if isinstance(metric_loss, torch.Tensor) else float(metric_loss))
        self.score.update(
            predictions,
            targets,
            loss=score_loss,
        )

        batch_primary_score = score_loss
        if self._primary_score_name() == "perplexity":
            batch_primary_score = float(math.exp(score_loss))

        return {
            "loss": float(loss.detach().item()),
            "num_samples": float(self._batch_size(batch)),
            "batch_primary_score": batch_primary_score,
        }

    def _resolve_device(self, requested: str) -> torch.device:
        req = requested.lower().strip()
        if req == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if req == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("training.device is set to cuda but CUDA is not available")
        return torch.device(req)

    def _move_batch_to_device(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        if isinstance(batch, list):
            return [self._move_batch_to_device(v) for v in batch]
        if isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(v) for v in batch)
        return batch

    def _batch_size(self, batch: Any) -> int:
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids")
            if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 1:
                return int(input_ids.shape[0])
            targets = batch.get("targets")
            if targets is None:
                targets = batch.get("labels")
            if isinstance(targets, torch.Tensor) and targets.ndim >= 1:
                return int(targets.shape[0])
        if isinstance(batch, torch.Tensor) and batch.ndim >= 1:
            return int(batch.shape[0])
        return 0

    def _run_dataloader(
        self,
        dataloader: Iterable[Any] | None,
        *,
        is_train: bool,
        optimizer: torch.optim.Optimizer | None,
        score_mode: str = "default",
        logger: Logger | None = None,
        training_mode: str | None = None,
        learner_id: int | None = None,
        epoch: int | None = None,
        split_name: str = "train",
    ) -> dict[str, Any]:
        if dataloader is None:
            return {
                "loss": 0.0,
                "num_batches": 0.0,
                "num_samples": 0.0,
                "score": 0.0,
                "score_name": self._primary_score_name(),
            }

        self.score.reset()
        total_loss = 0.0
        num_samples = 0
        iterable = tqdm.tqdm(dataloader, desc="Processing batches")
        for batch_idx, batch in enumerate(iterable, start=1):
            batch_stats = self._process_batch(batch, is_train=is_train, optimizer=optimizer, score_mode=score_mode)
            total_loss += batch_stats["loss"]
            num_samples += int(batch_stats["num_samples"])
            if (
                logger is not None
                and is_train
                and training_mode == "boosting"
                and learner_id is not None
                and epoch is not None
            ):
                score_name = self._primary_score_name()
                logger.info(
                    "mode=boosting phase=%s learner=%d epoch=%d batch=%d stage_mse_loss=%.6f %s=%.6f",
                    split_name,
                    learner_id,
                    epoch,
                    batch_idx,
                    batch_stats["loss"],
                    score_name,
                    batch_stats["batch_primary_score"],
                )

        primary_score_name = self._primary_score_name()
        score_metrics = self.score.compute()
        primary_score = self._extract_primary_score(score_metrics, primary_score_name)

        # Save losses and scores to results for summary stats
        num_batches = len(dataloader)
        if num_batches == 0:
            return {
                "loss": 0.0,
                "num_batches": 0.0,
                "num_samples": float(num_samples),
                "score": float(primary_score),
                "score_name": primary_score_name,
            }
        return {
            "loss": total_loss / num_batches,
            "num_batches": float(num_batches),
            "num_samples": float(num_samples),
            "score": float(primary_score),
            "score_name": primary_score_name,
        }

    def _train_loop(
        self,
        *,
        train_dataloader: Iterable[Any] | None,
        val_dataloader: Iterable[Any] | None,
        epochs: int,
        eval_step: int,
        optimizer: torch.optim.Optimizer,
        logger: Logger,
        results: dict[str, Any],
        training_mode: str,
        learner_id: int | None,
    ) -> None:
        for epoch in range(1, epochs + 1):
            train_start = perf_counter()
            train_stats = self._run_dataloader(
                train_dataloader,
                is_train=True,
                optimizer=optimizer,
                score_mode="pre_alpha" if training_mode == "boosting" else "default",
                logger=logger,
                training_mode=training_mode,
                learner_id=learner_id,
                epoch=epoch,
                split_name="train",
            )
            train_duration = perf_counter() - train_start
            train_record = {
                "epoch": epoch,
                "split": "train",
                "training_mode": training_mode,
                "learner_id": learner_id,
                "num_samples": int(train_stats["num_samples"]),
                "num_batches": int(train_stats["num_batches"]),
                "avg_loss": train_stats["loss"],
                "avg_score": train_stats["score"],
                "score_name": train_stats["score_name"],
                "duration_sec": train_duration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._append_split_record(results, "train", train_record)

            if learner_id is None:
                logger.info(
                    "mode=%s epoch=%s split=train avg_loss=%.6f avg_score=%.6f score=%s samples=%d batches=%d time=%.2fs",
                    training_mode,
                    epoch,
                    train_stats["loss"],
                    train_stats["score"],
                    train_stats["score_name"],
                    int(train_stats["num_samples"]),
                    int(train_stats["num_batches"]),
                    train_duration,
                )
            else:
                if training_mode == "boosting":
                    logger.info(
                        "mode=boosting phase=train learner=%d epoch=%s avg_stage_mse=%.6f avg_%s=%.6f samples=%d batches=%d time=%.2fs",
                        learner_id,
                        epoch,
                        train_stats["loss"],
                        train_stats["score_name"],
                        train_stats["score"],
                        int(train_stats["num_samples"]),
                        int(train_stats["num_batches"]),
                        train_duration,
                    )
                else:
                    logger.info(
                        "mode=%s learner=%d epoch=%s split=train avg_loss=%.6f avg_score=%.6f score=%s samples=%d batches=%d time=%.2fs",
                        training_mode,
                        learner_id,
                        epoch,
                        train_stats["loss"],
                        train_stats["score"],
                        train_stats["score_name"],
                        int(train_stats["num_samples"]),
                        int(train_stats["num_batches"]),
                        train_duration,
                    )

            if eval_step > 0 and epoch % eval_step == 0:
                val_start = perf_counter()
                val_stats = self._run_dataloader(
                    val_dataloader,
                    is_train=False,
                    optimizer=None,
                    score_mode="pre_alpha" if training_mode == "boosting" else "default",
                    split_name="val-pre-alpha",
                )
                val_duration = perf_counter() - val_start
                val_record = {
                    "epoch": epoch,
                    "split": "val",
                    "training_mode": training_mode,
                    "learner_id": learner_id,
                    "num_samples": int(val_stats["num_samples"]),
                    "num_batches": int(val_stats["num_batches"]),
                    "avg_loss": val_stats["loss"],
                    "avg_score": val_stats["score"],
                    "score_name": val_stats["score_name"],
                    "duration_sec": val_duration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._append_split_record(results, "val", val_record)

                if learner_id is None:
                    logger.info(
                        "mode=%s epoch=%s split=val avg_loss=%.6f avg_score=%.6f score=%s samples=%d batches=%d time=%.2fs",
                        training_mode,
                        epoch,
                        val_stats["loss"],
                        val_stats["score"],
                        val_stats["score_name"],
                        int(val_stats["num_samples"]),
                        int(val_stats["num_batches"]),
                        val_duration,
                    )
                else:
                    if training_mode == "boosting":
                        logger.info(
                            "mode=boosting phase=val-pre-alpha learner=%d epoch=%s avg_stage_mse=%.6f avg_%s=%.6f samples=%d batches=%d time=%.2fs",
                            learner_id,
                            epoch,
                            val_stats["loss"],
                            val_stats["score_name"],
                            val_stats["score"],
                            int(val_stats["num_samples"]),
                            int(val_stats["num_batches"]),
                            val_duration,
                        )
                    else:
                        logger.info(
                            "mode=%s learner=%d epoch=%s split=val avg_loss=%.6f avg_score=%.6f score=%s samples=%d batches=%d time=%.2fs",
                            training_mode,
                            learner_id,
                            epoch,
                            val_stats["loss"],
                            val_stats["score"],
                            val_stats["score_name"],
                            int(val_stats["num_samples"]),
                            int(val_stats["num_batches"]),
                            val_duration,
                        )

    def _append_split_record(self, results: dict[str, Any], split: str, record: dict[str, Any]) -> None:
        split_node = results[split]
        split_node["records"].append(record)
        split_node["losses"].append(record["avg_loss"])

        score_name = record["score_name"]
        scores = split_node["scores"]
        if score_name not in scores:
            scores[score_name] = []
        scores[score_name].append(record["avg_score"])

        split_node["metadata"]["num_samples"] = max(
            int(split_node["metadata"].get("num_samples", 0)),
            int(record["num_samples"]),
        )

    def _primary_score_name(self) -> str:
        metrics = self.config.get("score", {}).get("params", {}).get("metrics", ["loss"])
        if isinstance(metrics, list) and metrics:
            return str(metrics[0])
        return "loss"

    def _extract_primary_score(self, metric_values: dict[str, Any], primary_score_name: str) -> float:
        if primary_score_name in metric_values:
            try:
                return float(metric_values[primary_score_name])
            except (TypeError, ValueError):
                return 0.0
        if metric_values:
            first_value = next(iter(metric_values.values()))
            try:
                return float(first_value)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _build_run_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_type = self.config.get("model", {}).get("type", "model")
        data_type = self.config.get("data", {}).get("type", "data")
        return f"{stamp}_{model_type}_{data_type}"

    def _resolve_results_path(self, run_id: str) -> Path:
        training_config = self.config["training"]
        results_dir = Path(training_config.get("results_dir", "results/runs"))
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / "metrics.json"

    def _collect_data_meta(self) -> dict[str, Any]:
        data_meta: dict[str, Any] = {}
        raw_meta = getattr(self.data_module, "meta", None)
        if isinstance(raw_meta, dict):
            data_meta.update(raw_meta)
        data_meta["block_size"] = self.config.get("data", {}).get("params", {}).get("block_size")
        train_tokens = getattr(self.data_module, "_train_tokens", None)
        val_tokens = getattr(self.data_module, "_val_tokens", None)
        if isinstance(train_tokens, torch.Tensor):
            data_meta["num_train_samples"] = int(train_tokens.numel())
        if isinstance(val_tokens, torch.Tensor):
            data_meta["num_val_samples"] = int(val_tokens.numel())
        return data_meta

    def _build_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        train_losses = results["train"].get("losses", [])
        val_losses = results["val"].get("losses", [])
        primary_score = self._primary_score_name()
        train_scores = results["train"].get("scores", {}).get(primary_score, [])
        val_scores = results["val"].get("scores", {}).get(primary_score, [])

        training_mode = results.get("metadata", {}).get("training_mode", "baseline")

        best_epoch = None
        best_metric_name = self.config["training"].get("early_stopping", {}).get("metric", "loss")
        best_metric_value = None
        if train_losses and training_mode != "boosting":
            best_metric_value = min(train_losses)
            best_epoch = train_losses.index(best_metric_value) + 1

        boosting_summary: dict[str, Any] = {}
        if training_mode == "boosting":
            boosting_summary = self._build_boosting_summary(results, primary_score)

        return {
            "avg_train_loss": float(mean(train_losses)) if train_losses else 0.0,
            "avg_train_score": float(mean(train_scores)) if train_scores else 0.0,
            "avg_val_loss": float(mean(val_losses)) if val_losses else 0.0,
            "avg_val_score": float(mean(val_scores)) if val_scores else 0.0,
            "best_epoch": best_epoch,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "num_epochs_ran": len(train_losses),
            "boosting_summary": boosting_summary,
        }

    def _build_boosting_summary(self, results: dict[str, Any], primary_score: str) -> dict[str, Any]:
        train_records = results.get("train", {}).get("records", [])
        val_records = results.get("val", {}).get("records", [])
        learner_ids = sorted(
            {
                int(record["learner_id"])
                for record in train_records + val_records
                if record.get("learner_id") is not None
            }
        )

        per_learner: list[dict[str, Any]] = []
        for learner_id in learner_ids:
            learner_train = [r for r in train_records if r.get("learner_id") == learner_id]
            learner_val = [r for r in val_records if r.get("learner_id") == learner_id]

            learner_train_post = [r for r in learner_train if bool(r.get("post_line_search", False))]
            learner_val_post = [r for r in learner_val if bool(r.get("post_line_search", False))]
            train_source = learner_train_post if learner_train_post else learner_train
            val_source = learner_val_post if learner_val_post else learner_val

            train_losses = [float(r["avg_loss"]) for r in train_source]
            train_scores = [float(r["avg_score"]) for r in train_source]
            val_losses = [float(r["avg_loss"]) for r in val_source]
            val_scores = [float(r["avg_score"]) for r in val_source]

            best_train_loss = min(train_losses) if train_losses else None
            best_train_epoch = None
            if best_train_loss is not None:
                best_train_epoch = next(
                    (int(r["epoch"]) for r in train_source if float(r["avg_loss"]) == best_train_loss),
                    None,
                )

            per_learner.append(
                {
                    "learner_id": learner_id,
                    "avg_train_loss": float(mean(train_losses)) if train_losses else 0.0,
                    "avg_train_score": float(mean(train_scores)) if train_scores else 0.0,
                    "avg_val_loss": float(mean(val_losses)) if val_losses else 0.0,
                    "avg_val_score": float(mean(val_scores)) if val_scores else 0.0,
                    "best_train_loss": best_train_loss,
                    "best_train_epoch": best_train_epoch,
                    "primary_score": primary_score,
                    "uses_post_line_search_metrics": bool(learner_train_post or learner_val_post),
                }
            )

        return {"per_learner": per_learner}

    def _write_results_json(self, path: Path, results: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=True, indent=2, default=str)
