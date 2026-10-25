"""
File: visualize_attention.py
What this file does:
- Loads the best checkpoint and plots, for each layer and head, how much the LAST token
  (the pointer) attends to every position of one validation sequence.

Why it is useful:
- On the "pointer" task the correct answer sits at the position the last token points to.
  If attention learned the task, some head should put most of its weight on that position.

Usage:
    python visualize_attention.py            # writes assets/attention.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # Headless backend: write files, no window.
import matplotlib.pyplot as plt  # Plotting.
import torch  # Core tensor library.

from dataset import DatasetConfig, SyntheticSequenceDataset  # Same data rules as training.
from model import ModelConfig, MultiHeadSelfAttention, TransformerForSequencePrediction  # Model.


def main() -> None:
    data_cfg = DatasetConfig()  # Same defaults as training.
    model_cfg = ModelConfig(vocab_size=data_cfg.vocab_size, seq_len=data_cfg.seq_len)

    model = TransformerForSequencePrediction(model_cfg)  # CPU is enough for one sample.
    state = torch.load(os.path.join("checkpoints", "best.pt"), map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()  # Disables dropout, so stored attention is the real one.

    val_ds = SyntheticSequenceDataset(
        size=1, seq_len=data_cfg.seq_len, vocab_size=data_cfg.vocab_size,
        seed=data_cfg.seed + 1, task=data_cfg.task,
    )
    tokens, target = val_ds[0]  # One validation example.
    pointer = int(tokens[-1])  # Position the last token points to.

    with torch.no_grad():
        pred = model(tokens.unsqueeze(0)).argmax(dim=-1).item()

    attn_layers = [m for m in model.modules() if isinstance(m, MultiHeadSelfAttention)]
    num_layers, num_heads = len(attn_layers), model_cfg.num_heads

    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 1.6 * num_layers), sharex=True)
    for layer_idx, (ax, attn) in enumerate(zip(axes, attn_layers)):
        last_row = attn.last_attn[0, :, -1, :]  # (H, N): last token's attention per head.
        ax.imshow(last_row.numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.axvline(pointer, color="red", linewidth=1.5)  # Correct position.
        ax.set_ylabel(f"layer {layer_idx}\nheads", fontsize=8)
        ax.set_yticks(range(num_heads))

    axes[-1].set_xticks(range(data_cfg.seq_len))
    axes[-1].set_xticklabels([str(int(t)) for t in tokens], fontsize=8)
    axes[-1].set_xlabel("input tokens (last token = pointer); red line = pointed position")
    fig.suptitle(f"Attention of the last token | pointer={pointer} target={int(target)} prediction={pred}")
    fig.tight_layout()

    os.makedirs("assets", exist_ok=True)
    fig.savefig(os.path.join("assets", "attention.png"), dpi=130)
    print(f"pointer={pointer} target={int(target)} prediction={pred} -> assets/attention.png")


if __name__ == "__main__":
    main()
