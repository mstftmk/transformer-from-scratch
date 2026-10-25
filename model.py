"""
File: model.py
What this file does:
- Implements a Transformer backbone from scratch in PyTorch.
- Includes Embedding, Positional Encoding, Self-Attention, FFN, LayerNorm.
- Provides a sequence prediction head on top of the backbone.

Why it is written this way:
- The code is explicit and readable to teach each component clearly.
- Pre-norm blocks improve gradient flow in deeper stacks.

Alternatives:
- Use torch.nn.MultiheadAttention for brevity.
- Use post-norm blocks (LayerNorm after residual) or RMSNorm.
- Use learned positional embeddings instead of sinusoidal.

Line-by-line guidance:
- Each line has a short inline comment to explain its role.
"""

from __future__ import annotations  # Enable forward type references.

import math  # For sqrt and trig functions.
from dataclasses import dataclass  # Lightweight config container.
from typing import Optional  # For optional mask types.

import torch  # Core tensor library.
import torch.nn as nn  # Neural network modules.


@dataclass
class ModelConfig:
    """Configuration for the Transformer model."""

    vocab_size: int = 50  # Vocabulary size (V).
    seq_len: int = 16  # Input sequence length (N).
    embed_dim: int = 64  # Embedding dimension (C).
    num_heads: int = 4  # Number of attention heads (H).
    num_layers: int = 4  # Depth of Transformer stack.
    ffn_hidden_dim: int = 256  # Hidden size for FFN (usually 4x C).
    dropout: float = 0.1  # Dropout probability.
    causal: bool = True  # Use causal mask for next-token prediction.


class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds fixed sinusoidal positional encodings to token embeddings.

    Why sinusoidal:
    - Allows extrapolation to longer sequences.
    - No extra parameters.
    """

    def __init__(self, max_len: int, embed_dim: int, dropout: float) -> None:
        super().__init__()  # Initialize nn.Module.
        self.dropout = nn.Dropout(dropout)  # Regularize embeddings.

        # Create positional encodings once; shape: (max_len, embed_dim).
        pe = torch.zeros(max_len, embed_dim)  # (N, C)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (N, 1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        )  # (C/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices.
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices.
        pe = pe.unsqueeze(0)  # (1, N, C) for broadcasting.
        self.register_buffer("pe", pe, persistent=False)  # Non-trainable buffer.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        x = x + self.pe[:, : x.size(1)]  # Add positional info.
        x = self.dropout(x)  # Dropout for regularization.
        return x  # (B, N, C)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention implemented from scratch.

    Key ideas:
    - Project input to queries, keys, values.
    - Compute scaled dot-product attention per head.
    - Concatenate heads and project back to embed_dim.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, causal: bool) -> None:
        super().__init__()  # Initialize nn.Module.
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"  # Sanity.

        self.embed_dim = embed_dim  # C
        self.num_heads = num_heads  # H
        self.head_dim = embed_dim // num_heads  # C/H
        self.scale = 1.0 / math.sqrt(self.head_dim)  # Scale for stability.
        self.causal = causal  # Whether to use causal mask.

        # Single projection for Q, K, V for efficiency.
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)  # (C -> 3C)
        self.attn_drop = nn.Dropout(dropout)  # Dropout on attention weights.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # Output projection.
        self.proj_drop = nn.Dropout(dropout)  # Dropout after projection.

        # Cache for causal mask to avoid re-allocating each step.
        self.register_buffer("causal_mask", None, persistent=False)  # Built on first use.

    def _get_causal_mask(self, n: int, device: torch.device) -> torch.Tensor:
        # Build or reuse a lower-triangular mask of shape (1, 1, N, N).
        if self.causal_mask is None or self.causal_mask.size(-1) < n:
            mask = torch.tril(torch.ones(n, n, device=device, dtype=torch.bool))  # (N, N)
            self.causal_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        return self.causal_mask[:, :, :n, :n]  # (1, 1, N, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        b, n, c = x.shape  # Unpack shapes for readability.

        # Project to Q, K, V in one go; shape: (B, N, 3C).
        qkv = self.qkv(x)

        # Reshape to separate heads and Q/K/V: (B, N, 3, H, Hd).
        qkv = qkv.view(b, n, 3, self.num_heads, self.head_dim)

        # Move dimensions to (3, B, H, N, Hd) for easy split.
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Split into q, k, v; each is (B, H, N, Hd).
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores: (B, H, N, N).
        # Memory note: this quadratic matrix is the main memory cost of attention.
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal masking prevents information leak from future tokens.
        if self.causal:
            mask = self._get_causal_mask(n, x.device)  # (1, 1, N, N)
            # Use dtype min for numerical stability in mixed precision.
            attn_scores = attn_scores.masked_fill(~mask, torch.finfo(attn_scores.dtype).min)

        # Stable softmax: subtract max to reduce overflow risk.
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, H, N, N)
        self.last_attn = attn_weights.detach()  # Kept for visualization (see visualize_attention.py).
        attn_weights = self.attn_drop(attn_weights)  # Regularize attention.

        # Attention output: (B, H, N, Hd).
        attn_out = torch.matmul(attn_weights, v)

        # Merge heads: (B, N, H, Hd) -> (B, N, C).
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, n, c)

        # Project back to embed_dim.
        out = self.out_proj(attn_out)
        out = self.proj_drop(out)  # Dropout after projection.

        return out  # (B, N, C)


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network (FFN).

    Structure: Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()  # Initialize nn.Module.
        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # (C -> H)
        self.act = nn.GELU()  # Smooth nonlinearity.
        self.drop1 = nn.Dropout(dropout)  # Dropout after activation.
        self.fc2 = nn.Linear(hidden_dim, embed_dim)  # (H -> C)
        self.drop2 = nn.Dropout(dropout)  # Dropout after second linear.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        x = self.fc1(x)  # (B, N, H)
        x = self.act(x)  # Nonlinear transform.
        x = self.drop1(x)  # Regularization.
        x = self.fc2(x)  # (B, N, C)
        x = self.drop2(x)  # Regularization.
        return x  # (B, N, C)


class TransformerBlock(nn.Module):
    """
    A single Transformer block with pre-norm architecture.

    Pre-norm helps gradient flow in deep stacks and stabilizes training.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int, dropout: float, causal: bool) -> None:
        super().__init__()  # Initialize nn.Module.
        self.ln1 = nn.LayerNorm(embed_dim)  # Normalization before attention.
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, causal)  # MHA.
        self.ln2 = nn.LayerNorm(embed_dim)  # Normalization before FFN.
        self.ffn = FeedForward(embed_dim, ffn_hidden_dim, dropout)  # FFN.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        # Residual connection preserves gradient flow and signal.
        x = x + self.attn(self.ln1(x))  # (B, N, C)
        # Second residual path through FFN.
        x = x + self.ffn(self.ln2(x))  # (B, N, C)
        return x  # (B, N, C)


class TransformerBackbone(nn.Module):
    """
    Stacks multiple Transformer blocks to form the backbone.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()  # Initialize nn.Module.
        self.cfg = cfg  # Store config for reference.

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)  # Token embeddings.
        self.pos_embed = SinusoidalPositionalEncoding(cfg.seq_len, cfg.embed_dim, cfg.dropout)  # Pos enc.
        self.drop = nn.Dropout(cfg.dropout)  # Extra dropout after embeddings.

        # Stack of Transformer blocks.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    ffn_hidden_dim=cfg.ffn_hidden_dim,
                    dropout=cfg.dropout,
                    causal=cfg.causal,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(cfg.embed_dim)  # Final LayerNorm.

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, N)
        x = self.token_embed(input_ids)  # (B, N, C)
        x = self.pos_embed(x)  # (B, N, C)
        x = self.drop(x)  # (B, N, C)

        # Pass through all blocks.
        for block in self.blocks:
            x = block(x)  # (B, N, C)

        x = self.ln_f(x)  # (B, N, C)
        return x  # (B, N, C)


class TransformerForSequencePrediction(nn.Module):
    """
    Full model: Transformer backbone + linear head for next-token prediction.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()  # Initialize nn.Module.
        self.backbone = TransformerBackbone(cfg)  # Core Transformer.
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size)  # Projection to vocab.

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, N)
        features = self.backbone(input_ids)  # (B, N, C)

        # Use the last position to predict the next token.
        last_token = features[:, -1, :]  # (B, C)

        logits = self.head(last_token)  # (B, V)
        return logits  # (B, V)
