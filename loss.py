"""
File: loss.py
What this file does:
- Defines the loss function used for sequence prediction.

Why it is written this way:
- Wraps CrossEntropyLoss in a small module for clarity and extensibility.

Alternatives:
- Label smoothing for improved generalization.
- Focal loss for imbalanced targets.

Line-by-line guidance:
- Each line includes a brief inline comment explaining its role.
"""

from __future__ import annotations  # Enable forward type references.

import torch  # Core tensor library.
import torch.nn as nn  # Neural network modules.


class SequencePredictionLoss(nn.Module):
    """Loss for predicting a single next token from a sequence."""

    def __init__(self) -> None:
        super().__init__()  # Initialize nn.Module.
        self.criterion = nn.CrossEntropyLoss()  # Standard classification loss.

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, V), targets: (B,)
        loss = self.criterion(logits, targets)  # Compute cross-entropy.
        return loss  # Scalar tensor.
