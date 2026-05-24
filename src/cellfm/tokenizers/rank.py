"""Rank-token tokenizer (Geneformer-style).

Per cell: sort detected (nonzero) genes by expression (descending),
take the top L, pad to L with PAD. Magnitudes are discarded — only the
gene ID at each rank position is preserved.

Vocab size: n_genes + N_SPECIAL (PAD, CLS, MASK).

Masking for MLM pretraining: 15% of non-special positions get replaced
with MASK_ID; targets are the original gene IDs.
"""

from __future__ import annotations

import numpy as np
import torch

from cellfm.tokenizers.base import (
    CLS_ID,
    MASK_ID,
    N_SPECIAL,
    PAD_ID,
    TokenizerConfig,
)


class RankTokenizer:
    name = "rank"

    def __init__(self, cfg: TokenizerConfig):
        self.cfg = cfg
        self.L = cfg.L
        self.add_cls = cfg.add_cls
        self.body_len = self.L - 1 if self.add_cls else self.L
        self.mask_ratio = cfg.mask_ratio

    @property
    def gene_vocab_size(self) -> int:
        return int(self.cfg.n_genes + N_SPECIAL)

    @property
    def value_vocab_size(self) -> int | None:
        return None

    def _encode_one(self, gene_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
        if gene_idx.shape[0] == 0:
            return np.full(self.body_len, PAD_ID, dtype=np.int64)
        # Sort by value descending (stable so ties keep gene-index order)
        order = np.argsort(-values, kind="stable")
        sorted_genes = gene_idx[order]
        topL = sorted_genes[: self.body_len]
        tokens = np.full(self.body_len, PAD_ID, dtype=np.int64)
        tokens[: topL.shape[0]] = topL + N_SPECIAL
        return tokens

    def encode_batch(
        self, items: list[dict], *, train: bool = True
    ) -> dict[str, torch.Tensor]:
        B = len(items)
        tokens = np.full((B, self.L), PAD_ID, dtype=np.int64)

        for i, item in enumerate(items):
            body = self._encode_one(
                item["gene_idx"].astype(np.int64),
                item["values"].astype(np.float32),
            )
            if self.add_cls:
                tokens[i, 0] = CLS_ID
                tokens[i, 1:] = body
            else:
                tokens[i, :] = body

        attention_mask = (tokens != PAD_ID).astype(np.bool_)

        # Build mask labels for MLM
        if train and self.mask_ratio > 0:
            targets, masked_tokens = _apply_mlm_mask(
                tokens, attention_mask, self.mask_ratio
            )
        else:
            targets = np.full_like(tokens, -100)
            masked_tokens = tokens

        labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
        return {
            "input_ids": torch.from_numpy(masked_tokens),
            "attention_mask": torch.from_numpy(attention_mask),
            "mlm_targets": torch.from_numpy(targets),  # -100 outside masked positions
            "labels": labels,
        }


def _apply_mlm_mask(
    tokens: np.ndarray, attention_mask: np.ndarray, mask_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Standard BERT-style masking: targets are -100 outside masked positions.

    Only positions that are non-PAD and non-CLS are eligible.
    Of the chosen positions: 80% MASK, 10% random gene, 10% unchanged.
    """
    rng = np.random.default_rng()
    targets = np.full_like(tokens, -100)
    masked_tokens = tokens.copy()

    eligible = attention_mask & (tokens != CLS_ID)
    B, L = tokens.shape
    for b in range(B):
        positions = np.where(eligible[b])[0]
        if positions.size == 0:
            continue
        k = max(1, int(round(mask_ratio * positions.size)))
        chosen = rng.choice(positions, size=k, replace=False)
        targets[b, chosen] = tokens[b, chosen]

        # Split chosen into 80/10/10
        roll = rng.random(size=k)
        mask_idx = chosen[roll < 0.8]
        rand_idx = chosen[(roll >= 0.8) & (roll < 0.9)]
        # keep_idx = chosen[roll >= 0.9]  # leave unchanged

        masked_tokens[b, mask_idx] = MASK_ID
        if rand_idx.size > 0:
            rand_vals = rng.integers(
                low=N_SPECIAL, high=tokens.max() + 1 if tokens.max() >= N_SPECIAL else N_SPECIAL + 1,
                size=rand_idx.size,
            )
            masked_tokens[b, rand_idx] = rand_vals
    return targets, masked_tokens
