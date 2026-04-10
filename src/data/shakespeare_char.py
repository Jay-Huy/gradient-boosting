from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.base import BaseDataModule


class CharSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, tokens: torch.Tensor, block_size: int) -> None:
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, self.tokens.numel() - self.block_size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x = self.tokens[index : index + self.block_size]
        y = self.tokens[index + 1 : index + 1 + self.block_size]
        return {"input_ids": x.long(), "targets": y.long()}


class ShakespeareCharDataModule(BaseDataModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.dataset_dir = Path(self.config.get("dataset_dir", "data/shakespeare_char"))
        self.input_file = self.dataset_dir / self.config.get("input_file", "input.txt")
        self.train_bin_file = self.dataset_dir / self.config.get("train_bin", "train.bin")
        self.val_bin_file = self.dataset_dir / self.config.get("val_bin", "val.bin")
        self.meta_file = self.dataset_dir / self.config.get("meta_file", "meta.pkl")

        self.block_size = int(self.config["block_size"])
        self.batch_size = int(self.config["batch_size"])
        self.num_workers = int(self.config.get("num_workers", 0))
        self.pin_memory = bool(self.config.get("pin_memory", False))
        self.val_split = float(self.config.get("val_split", 0.1))
        self.download_if_missing = bool(self.config.get("download_if_missing", False))
        self.reprepare = bool(self.config.get("reprepare", False))

        self._train_tokens: torch.Tensor | None = None
        self._val_tokens: torch.Tensor | None = None
        self.meta: dict[str, Any] = {}

    def encode(self, text: str) -> list[int]:
        stoi = self.meta["stoi"]
        return [stoi[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        itos = self.meta["itos"]
        return "".join(itos[int(idx)] for idx in token_ids)

    def load_data(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        if self.reprepare or not self._prepared_files_exist():
            self._prepare_files()

        with self.meta_file.open("rb") as handle:
            self.raw_data = pickle.load(handle)

        train_np = np.fromfile(self.train_bin_file, dtype=np.uint16).astype(np.int64)
        val_np = np.fromfile(self.val_bin_file, dtype=np.uint16).astype(np.int64)
        self._train_tokens = torch.from_numpy(train_np)
        self._val_tokens = torch.from_numpy(val_np)

        self.meta = self._get_meta() # Get metadata for vocab size and encoding/decoding

    def setup_dataloaders(self) -> None:
        if self._train_tokens is None or self._val_tokens is None:
            raise RuntimeError("Tokens are not loaded. Call load_data() before setup_dataloaders().")

        train_dataset = CharSequenceDataset(self._train_tokens, self.block_size)
        val_dataset = CharSequenceDataset(self._val_tokens, self.block_size)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _prepared_files_exist(self) -> bool:
        return self.train_bin_file.exists() and self.val_bin_file.exists() and self.meta_file.exists()

    def _prepare_files(self) -> None:
        if not self.input_file.exists():
            if not self.download_if_missing:
                raise FileNotFoundError(
                    f"Missing dataset file: {self.input_file}. Set data.params.download_if_missing=true to auto-download."
                )
            self._download_input_text()

        text = self.input_file.read_text(encoding="utf-8")
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        train_cutoff = int(len(text) * (1.0 - self.val_split))
        train_text = text[:train_cutoff]
        val_text = text[train_cutoff:]

        train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
        val_ids = np.array([stoi[c] for c in val_text], dtype=np.uint16)
        train_ids.tofile(self.train_bin_file)
        val_ids.tofile(self.val_bin_file)

        meta = {
            "vocab_size": len(chars),
            "stoi": stoi,
            "itos": itos,
        }
        with self.meta_file.open("wb") as handle:
            pickle.dump(meta, handle)

    def _download_input_text(self) -> None:
        url = self.config.get(
            "dataset_url",
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        )
        with urlopen(url) as response:
            data = response.read().decode("utf-8")
        self.input_file.write_text(data, encoding="utf-8")

    def _get_meta(self) -> dict[str, Any]:
        if isinstance(self.raw_data, dict):
            return self.raw_data
        if not self.meta_file.exists():
            raise RuntimeError("Meta file not found. Call load_data() first or prepare dataset files.")
        with self.meta_file.open("rb") as handle:
            self.raw_data = pickle.load(handle)
        if not isinstance(self.raw_data, dict):
            raise RuntimeError("Invalid metadata format in meta.pkl")
        return self.raw_data
