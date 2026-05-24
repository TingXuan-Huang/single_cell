"""Weighted EmbeddingBag tokenizer.

Produces the (indices, offsets, per_sample_weights) tuple that
`torch.nn.EmbeddingBag(mode='sum')` expects. This is the same operation as
sparse linear (see memory/LEARNINGS.md#sparse-linear-equals-weighted-bag);
we use EmbeddingBag because it is efficient and Allen-scale-friendly.

Vocab size: n_genes (no special tokens; reconstruction objective handles
masking by zeroing values, not by a MASK id).

For the reconstruction self-supervised objective:
- Pick mask_ratio fraction of each cell's nonzero genes.
- Zero them out in the per_sample_weights.
- Predict the original (log1p-normalized) values at the masked positions
  from the bag embedding.
"""

from __future__ import annotations

import numpy as np
import torch

from cellfm.tokenizers.base import TokenizerConfig


class EmbeddingBagTokenizer:
    name = "embedding_bag"

    def __init__(self, cfg: TokenizerConfig):
        self.cfg = cfg

    @property
    def gene_vocab_size(self) -> int:
        return int(self.cfg.n_genes)

    @property
    def value_vocab_size(self) -> int | None:
        return None

    def encode_batch(
        self, items: list[dict], *, train: bool = True
    ) -> dict[str, torch.Tensor]:
        B = len(items)
        offsets = np.zeros(B, dtype=np.int64)
        all_indices: list[np.ndarray] = []
        all_weights: list[np.ndarray] = []

        # Track masked positions for the reconstruction objective.
        masked_gene_lists: list[np.ndarray] = []
        masked_value_lists: list[np.ndarray] = []
        masked_cell_idx: list[np.ndarray] = []

        cursor = 0
        rng = np.random.default_rng()
        for i, item in enumerate(items):
            g = item["gene_idx"].astype(np.int64)
            v = item["values"].astype(np.float32)
            if self.cfg.log1p_normalize:
                total = v.sum()
                if total > 0:
                    v = v * (self.cfg.target_sum / total)
                v = np.log1p(v)

            v_input = v.copy()
            if train and self.cfg.mask_ratio > 0 and g.shape[0] > 0:
                k = max(1, int(round(self.cfg.mask_ratio * g.shape[0])))
                mask_pos = rng.choice(g.shape[0], size=k, replace=False)
                masked_gene_lists.append(g[mask_pos])
                masked_value_lists.append(v[mask_pos])
                masked_cell_idx.append(np.full(k, i, dtype=np.int64))
                v_input[mask_pos] = 0.0

            offsets[i] = cursor
            all_indices.append(g)
            all_weights.append(v_input)
            cursor += g.shape[0]

        indices = (
            np.concatenate(all_indices) if all_indices else np.zeros(0, dtype=np.int64)
        )
        weights = (
            np.concatenate(all_weights) if all_weights else np.zeros(0, dtype=np.float32)
        )

        masked_genes = (
            np.concatenate(masked_gene_lists) if masked_gene_lists else np.zeros(0, np.int64)
        )
        masked_values = (
            np.concatenate(masked_value_lists) if masked_value_lists else np.zeros(0, np.float32)
        )
        masked_cells = (
            np.concatenate(masked_cell_idx) if masked_cell_idx else np.zeros(0, np.int64)
        )

        labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
        return {
            "indices": torch.from_numpy(indices),
            "offsets": torch.from_numpy(offsets),
            "per_sample_weights": torch.from_numpy(weights),
            "labels": labels,
            "masked_genes": torch.from_numpy(masked_genes),
            "masked_values": torch.from_numpy(masked_values),
            "masked_cells": torch.from_numpy(masked_cells),
        }
