"""HVG dense input encoder.

Output: (B, n_hvg) float32 of log1p-normalized expression on the HVG subset.
Supervised baseline. Smallest and cheapest of the four encoders.

Param count for this head is dominated by Linear(n_hvg -> d_model). At
n_hvg=2000, d_model=128 this is ~250k params, well within the 1M body budget.
"""

from __future__ import annotations

import numpy as np
import torch

from cellfm.tokenizers.base import Tokenizer, TokenizerConfig


class HVGDenseTokenizer:
    name = "hvg_dense"

    def __init__(self, cfg: TokenizerConfig):
        if cfg.hvg_indices is None or len(cfg.hvg_indices) == 0:
            raise ValueError("HVGDenseTokenizer requires cfg.hvg_indices to be non-empty.")
        self.cfg = cfg
        self.hvg = np.asarray(cfg.hvg_indices, dtype=np.int64)
        # Map gene_idx -> position-in-hvg (or -1)
        self._gene_to_pos = np.full(cfg.n_genes, -1, dtype=np.int64)
        self._gene_to_pos[self.hvg] = np.arange(len(self.hvg), dtype=np.int64)

    @property
    def gene_vocab_size(self) -> int:
        # No tokens — return the HVG dimensionality for symmetry.
        return int(len(self.hvg))

    @property
    def value_vocab_size(self) -> int | None:
        return None

    def encode_batch(
        self, items: list[dict], *, train: bool = True
    ) -> dict[str, torch.Tensor]:
        B = len(items)
        H = len(self.hvg)
        X = np.zeros((B, H), dtype=np.float32)

        for i, item in enumerate(items):
            g = item["gene_idx"]
            v = item["values"]
            pos = self._gene_to_pos[g]
            keep = pos >= 0
            if keep.any():
                X[i, pos[keep]] = v[keep]

        if self.cfg.log1p_normalize:
            # Per-cell CPM-ish normalize, then log1p.
            total = X.sum(axis=1, keepdims=True)
            total = np.maximum(total, 1.0)
            X = X * (self.cfg.target_sum / total)
            np.log1p(X, out=X)

        labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
        return {
            "x_dense": torch.from_numpy(X),
            "labels": labels,
        }
