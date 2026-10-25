"""
File: dataset.py
What this file does:
- Defines a small, realistic synthetic dataset for sequence prediction.
- Provides a helper to build PyTorch DataLoaders.

Why it is written this way:
- Synthetic data keeps the project lightweight and fully reproducible.
- Pre-generating samples makes training deterministic and fast.

Alternatives:
- Use a real dataset (e.g., WikiText, IMDB) with a tokenizer.
- Generate samples on-the-fly to reduce RAM (slower but memory-light).

Line-by-line guidance:
- Each line includes a brief inline comment explaining its role.
"""

from __future__ import annotations  # Enable forward type references.

from dataclasses import dataclass  # Lightweight config container.
from typing import Tuple  # Type hints for clarity.

import torch  # Core tensor library.
from torch.utils.data import Dataset, DataLoader  # Dataset and batching.


@dataclass
class DatasetConfig:
    """Configuration for the synthetic dataset."""

    train_size: int = 20000  # Number of training samples.
    val_size: int = 1000  # Number of validation samples.
    seq_len: int = 16  # Length of input sequences (N).
    vocab_size: int = 50  # Size of discrete token vocabulary (V).
    seed: int = 1234  # Seed for reproducible data.
    task: str = "pointer"  # "pointer" (content-addressed lookup) or "mod_sum" ((last + second_last) % V).
    batch_size: int = 64  # Batch size for DataLoaders.
    num_workers: int = 0  # Use 0 for maximal portability.


class SyntheticSequenceDataset(Dataset):
    """
    A tiny but structured sequence prediction dataset.

    Tasks:
    - "pointer": the last token is an index k in [0, N-2]; the target is the token at position k.
      The model must read the last token, then attend to the position it points to. This exercises
      exactly what self-attention and positional encoding are for, and it is learnable in a few epochs.
    - "mod_sum": target = (last + second_last) % V. This looks simple but is a classic "grokking"
      problem: a small model needs thousands of epochs and strong weight decay before it generalizes,
      so with the default budget it stays near chance (1 / V).
    """

    def __init__(self, size: int, seq_len: int, vocab_size: int, seed: int, task: str = "pointer") -> None:
        self.size = size  # Total number of samples.
        self.seq_len = seq_len  # Sequence length (N).
        self.vocab_size = vocab_size  # Vocabulary size (V).

        # Deterministic generator so results are reproducible across runs.
        g = torch.Generator().manual_seed(seed)

        # Pre-generate inputs for speed and determinism.
        # Shape: (size, seq_len) -> (B, N).
        self.inputs = torch.randint(
            low=0,
            high=vocab_size,
            size=(size, seq_len),
            generator=g,
            dtype=torch.long,
        )

        if task == "pointer":
            # Last token is a pointer into the first N-1 positions.
            pointers = torch.randint(0, seq_len - 1, (size,), generator=g)  # (B,)
            self.inputs[:, -1] = pointers  # Requires vocab_size >= seq_len - 1.
            self.targets = self.inputs[torch.arange(size), pointers]  # Token at the pointed position.
        elif task == "mod_sum":
            # Compute targets using the rule; shape: (size,) -> (B,).
            last = self.inputs[:, -1]  # Last token of each sequence.
            second_last = self.inputs[:, -2]  # Second-to-last token.
            self.targets = (last + second_last) % vocab_size  # Next-token label.
        else:
            raise ValueError(f"unknown task: {task}")

    def __len__(self) -> int:
        return self.size  # Number of samples in the dataset.

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return input sequence and target token.
        return self.inputs[idx], self.targets[idx]


def build_dataloaders(cfg: DatasetConfig) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders."""

    train_ds = SyntheticSequenceDataset(
        size=cfg.train_size,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        seed=cfg.seed,
        task=cfg.task,
    )

    val_ds = SyntheticSequenceDataset(
        size=cfg.val_size,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        seed=cfg.seed + 1,  # Different seed for validation set.
        task=cfg.task,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,  # Shuffle to reduce batch correlation.
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),  # Faster H2D transfers on CUDA only.
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader  # Return both loaders.
