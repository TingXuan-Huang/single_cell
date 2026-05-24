"""Value-bin tokenizer (scGPT-style).

Per cell: take detected (nonzero) genes, log1p-normalize values to a fixed
target_sum, bin nonzero values into K cell-relative quantile bins
(default K=51 per locked decision in notes/pipeline_v1_plan.md), order genes
by binned value descending, take top L, pad with PAD.

Each position emits two tokens:
- gene_token: gene ID in {PAD=0, CLS=1, MASK=2, gene+N_SPECIAL...}
- value_token: bin ID in {PAD=0, MASK=1, bin+N_VALUE_SPECIAL...}

Masking for MLM pretraining: 15% of non-special positions get masked.
Both gene_token and value_token at masked positions are set to MASK / VAL_MASK.
Targets are the original gene ID and bin ID.
"""

from __future__ import annotations

import numpy as np
import torch

from cellfm.tokenizers.base import (
    CLS_ID,
    MASK_ID,
    N_SPECIAL,
    N_VALUE_SPECIAL,
    PAD_ID,
    VAL_MASK_ID,
    VAL_PAD_ID,
    TokenizerConfig,
)


class ValueBinTokenizer:
    name = "value_bin"

    def __init__(self, cfg: TokenizerConfig):
        self.cfg = cfg
        self.L = cfg.L
        self.add_cls = cfg.add_cls
        self.body_len = self.L - 1 if self.add_cls else self.L
        self.n_bins = cfg.n_bins
        self.mask_ratio = cfg.mask_ratio

    @property
    def gene_vocab_size(self) -> int:
        return int(self.cfg.n_genes + N_SPECIAL)

    @property
    def value_vocab_size(self) -> int | None:
        return int(self.n_bins + N_VALUE_SPECIAL)

    def _bin_values(self, values: np.ndarray) -> np.ndarray:
        """scGPT-style cell-relative quantile binning of nonzero values.

        Returns integers in [0, n_bins-1].
        """
        if values.size == 0:
            return np.zeros(0, dtype=np.int64)
        # Log1p-normalize within cell
        v = values.astype(np.float64)
        total = v.sum()
        if total > 0:
            v = v * (self.cfg.target_sum / total)
        v = np.log1p(v)

        # Rank-based binning so duplicates fall in the same bin
        order = np.argsort(v, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(v.size)
        bins = np.floor(ranks * self.n_bins / max(v.size, 1)).astype(np.int64)
        bins = np.clip(bins, 0, self.n_bins - 1)
        return bins

    def _encode_one(
        self, gene_idx: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if gene_idx.shape[0] == 0:
            return (
                np.full(self.body_len, PAD_ID, dtype=np.int64),
                np.full(self.body_len, VAL_PAD_ID, dtype=np.int64),
            )
        bins = self._bin_values(values)
        # Sort by bin descending so high-expression tokens land first
        order = np.argsort(-bins, kind="stable")
        sorted_genes = gene_idx[order][: self.body_len]
        sorted_bins = bins[order][: self.body_len]

        g_tok = np.full(self.body_len, PAD_ID, dtype=np.int64)
        v_tok = np.full(self.body_len, VAL_PAD_ID, dtype=np.int64)
        g_tok[: sorted_genes.shape[0]] = sorted_genes + N_SPECIAL
        v_tok[: sorted_bins.shape[0]] = sorted_bins + N_VALUE_SPECIAL
        return g_tok, v_tok

    def encode_batch(
        self, items: list[dict], *, train: bool = True
    ) -> dict[str, torch.Tensor]:
        B = len(items)
        g_tokens = np.full((B, self.L), PAD_ID, dtype=np.int64)
        v_tokens = np.full((B, self.L), VAL_PAD_ID, dtype=np.int64)

        for i, item in enumerate(items):
            g_body, v_body = self._encode_one(
                item["gene_idx"].astype(np.int64),
                item["values"].astype(np.float32),
            )
            if self.add_cls:
                g_tokens[i, 0] = CLS_ID
                g_tokens[i, 1:] = g_body
                v_tokens[i, 0] = VAL_PAD_ID
                v_tokens[i, 1:] = v_body
            else:
                g_tokens[i, :] = g_body
                v_tokens[i, :] = v_body

        attention_mask = (g_tokens != PAD_ID).astype(np.bool_)

        if train and self.mask_ratio > 0:
            g_targets, v_targets, g_in, v_in = _apply_joint_mlm_mask(
                g_tokens, v_tokens, attention_mask, self.mask_ratio
            )
        else:
            g_targets = np.full_like(g_tokens, -100)
            v_targets = np.full_like(v_tokens, -100)
            g_in, v_in = g_tokens, v_tokens

        labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
        return {
            "gene_ids": torch.from_numpy(g_in),
            "value_ids": torch.from_numpy(v_in),
            "attention_mask": torch.from_numpy(attention_mask),
            "mlm_gene_targets": torch.from_numpy(g_targets),
            "mlm_value_targets": torch.from_numpy(v_targets),
            "labels": labels,
        }


def _apply_joint_mlm_mask(
    g_tokens: np.ndarray,
    v_tokens: np.ndarray,
    attention_mask: np.ndarray,
    mask_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mask gene and value tokens jointly at the same positions."""
    rng = np.random.default_rng()
    g_targets = np.full_like(g_tokens, -100)
    v_targets = np.full_like(v_tokens, -100)
    g_in = g_tokens.copy()
    v_in = v_tokens.copy()

    eligible = attention_mask & (g_tokens != CLS_ID)
    B = g_tokens.shape[0]
    for b in range(B):
        positions = np.where(eligible[b])[0]
        if positions.size == 0:
            continue
        k = max(1, int(round(mask_ratio * positions.size)))
        chosen = rng.choice(positions, size=k, replace=False)
        g_targets[b, chosen] = g_tokens[b, chosen]
        v_targets[b, chosen] = v_tokens[b, chosen]

        roll = rng.random(size=k)
        mask_idx = chosen[roll < 0.8]
        # 10% random replacement, 10% keep
        rand_idx = chosen[(roll >= 0.8) & (roll < 0.9)]

        g_in[b, mask_idx] = MASK_ID
        v_in[b, mask_idx] = VAL_MASK_ID
        if rand_idx.size > 0:
            g_in[b, rand_idx] = rng.integers(
                low=N_SPECIAL,
                high=max(g_tokens.max() + 1, N_SPECIAL + 1),
                size=rand_idx.size,
            )
            v_in[b, rand_idx] = rng.integers(
                low=2,
                high=max(v_tokens.max() + 1, 3),
                size=rand_idx.size,
            )
    return g_targets, v_targets, g_in, v_in
