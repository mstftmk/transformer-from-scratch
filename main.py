"""
File: main.py
What this file does:
- Entry point that wires dataset, model, loss, optimizer, scheduler, trainer.

Why it is written this way:
- Keeps configuration in one place and makes the pipeline explicit.

Alternatives:
- Use a config file (YAML/JSON) or argument parsing (argparse).
- Use experiment tracking tools (Weights & Biases, MLflow).

Line-by-line guidance:
- Each line includes a brief inline comment explaining its role.
"""

from __future__ import annotations  # Enable forward type references.

import logging  # Logging for training output.
import random  # Python RNG.

import numpy as np  # Numpy RNG.
import torch  # Core tensor library.

from dataset import DatasetConfig, build_dataloaders  # Data utilities.
from loss import SequencePredictionLoss  # Loss module.
from model import ModelConfig, TransformerForSequencePrediction  # Model module.
from trainer import TrainConfig, Trainer  # Training utilities.


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across libraries."""

    random.seed(seed)  # Python RNG.
    np.random.seed(seed)  # Numpy RNG.
    torch.manual_seed(seed)  # Torch CPU RNG.
    torch.cuda.manual_seed_all(seed)  # Torch GPU RNG.


def main() -> None:
    # Logging setup.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Configs.
    data_cfg = DatasetConfig()  # Dataset config.
    model_cfg = ModelConfig(vocab_size=data_cfg.vocab_size, seq_len=data_cfg.seq_len)  # Model config.
    train_cfg = TrainConfig()  # Training config.

    # Reproducibility.
    set_seed(42)

    # Device selection.
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU.
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU.
    else:
        device = torch.device("cpu")  # Fallback.
    logging.info(f"device={device}")

    # Data.
    train_loader, val_loader = build_dataloaders(data_cfg)

    # Model.
    model = TransformerForSequencePrediction(model_cfg).to(device)

    # Loss.
    loss_fn = SequencePredictionLoss()

    # Optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    # Scheduler (cosine decay over epochs).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.epochs,
    )

    # Trainer.
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=train_cfg,
    )

    # Train.
    trainer.fit()


if __name__ == "__main__":
    main()  # Run the training pipeline.
