"""Pre-norm transformer encoder body.

Small enough to be readable; not a copy of a HF block. Used by the rank-token
and value-bin transformers; the HVG-dense and EmbeddingBag heads bypass it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BodyConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    max_len: int = 2048


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        # x: (B, L, D), attn_mask: (B, L) bool, True = valid position (attend to it).
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # -> (B, n_heads, L, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # SDPA boolean attn_mask convention: True = include this (q,k) pair in attention.
        # We only do KEY-side padding masking, broadcast over query positions.
        sdpa_mask = None
        if attn_mask is not None:
            sdpa_mask = attn_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int, dropout: float):
        super().__init__()
        hidden = d_model * ffn_mult
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: BodyConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.ffn_mult, cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of pre-norm transformer blocks."""

    def __init__(self, cfg: BodyConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.norm(x)
