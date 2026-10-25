"""
File: trainer.py
What this file does:
- Implements the training and evaluation loops.
- Handles logging, checkpointing, and gradient clipping.

Why it is written this way:
- Centralizes training logic for readability and reuse.
- Uses a Trainer class to keep responsibilities clear.

Alternatives:
- Use a high-level trainer (PyTorch Lightning, Accelerate).
- Implement a functional training loop in main.py.

Line-by-line guidance:
- Each line includes a brief inline comment explaining its role.
"""

from __future__ import annotations  # Enable forward type references.

import logging  # Standard logging.
import os  # File system utilities.
from dataclasses import dataclass  # Lightweight config container.
from typing import Tuple  # Type hints.

import torch  # Core tensor library.
from torch.utils.data import DataLoader  # Dataloader type.


@dataclass
class TrainConfig:
    """Configuration for training."""

    epochs: int = 20  # Number of training epochs.
    lr: float = 1e-3  # Learning rate.
    weight_decay: float = 0.01  # AdamW weight decay.
    max_grad_norm: float = 1.0  # Gradient clipping threshold.
    log_interval: int = 100  # Steps between logs.
    ckpt_dir: str = "checkpoints"  # Directory for checkpoints.
    ckpt_interval: int = 1  # Save checkpoint every N epochs.
    use_amp: bool = False  # Mixed precision training.


class Trainer:
    """
    Orchestrates training and evaluation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: TrainConfig,
    ) -> None:
        self.model = model  # Model to train.
        self.loss_fn = loss_fn  # Loss function.
        self.optimizer = optimizer  # Optimizer.
        self.scheduler = scheduler  # Learning rate scheduler.
        self.train_loader = train_loader  # Training data loader.
        self.val_loader = val_loader  # Validation data loader.
        self.device = device  # CPU or GPU.
        self.cfg = cfg  # Training configuration.

        self.scaler = torch.amp.GradScaler(device.type, enabled=cfg.use_amp)  # AMP scaler.

        os.makedirs(cfg.ckpt_dir, exist_ok=True)  # Ensure checkpoint dir exists.

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], train: bool) -> Tuple[torch.Tensor, int]:
        # One forward pass, returns loss and the number of correct predictions.
        inputs, targets = batch  # Unpack batch.
        inputs = inputs.to(self.device)  # Move to device.
        targets = targets.to(self.device)  # Move to device.

        # Forward pass with optional AMP.
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=self.device.type, enabled=self.cfg.use_amp):
                logits = self.model(inputs)  # (B, V)
                loss = self.loss_fn(logits, targets)  # Scalar.

        correct = (logits.argmax(dim=-1) == targets).sum().item()  # Top-1 hits in this batch.
        return loss, correct  # Return loss tensor and hit count.

    def train_epoch(self, epoch: int) -> float:
        # Train for one epoch.
        self.model.train()  # Set train mode.
        total_loss = 0.0  # Accumulate loss for reporting.

        for step, batch in enumerate(self.train_loader, start=1):
            # Forward + backward.
            loss, _ = self._step(batch, train=True)

            # Backward pass with AMP support.
            self.optimizer.zero_grad(set_to_none=True)  # Clear gradients.
            self.scaler.scale(loss).backward()  # Backprop.

            # Gradient clipping for numerical stability.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

            self.scaler.step(self.optimizer)  # Optimizer step.
            self.scaler.update()  # Update scaler.

            total_loss += loss.item()  # Accumulate loss.

            if step % self.cfg.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]  # Current LR.
                logging.info(f"epoch={epoch} step={step} loss={loss.item():.4f} lr={lr:.6f}")

        self.scheduler.step()  # Step LR scheduler per epoch.

        avg_loss = total_loss / max(1, len(self.train_loader))  # Mean loss.
        return avg_loss  # Return average.

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        # Evaluate on validation set.
        self.model.eval()  # Set eval mode.
        total_loss = 0.0  # Accumulate loss.
        total_correct = 0  # Accumulate correct predictions.

        for batch in self.val_loader:
            loss, correct = self._step(batch, train=False)  # Forward only.
            total_loss += loss.item()
            total_correct += correct

        avg_loss = total_loss / max(1, len(self.val_loader))  # Mean loss.
        accuracy = total_correct / max(1, len(self.val_loader.dataset))  # Top-1 accuracy.
        return avg_loss, accuracy  # Return averages.

    def save_checkpoint(self, epoch: int, best: bool = False) -> None:
        # Save model, optimizer, scheduler states.
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
        }

        filename = "best.pt" if best else f"epoch_{epoch:03d}.pt"
        path = os.path.join(self.cfg.ckpt_dir, filename)
        torch.save(state, path)

    def fit(self) -> None:
        # Full training loop.
        best_val = float("inf")  # Track best validation loss.

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch(epoch)  # Train.
            val_loss, val_acc = self.evaluate()  # Validate.

            logging.info(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            # Save checkpoints periodically.
            if epoch % self.cfg.ckpt_interval == 0:
                self.save_checkpoint(epoch, best=False)

            # Save best model.
            if val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint(epoch, best=True)
