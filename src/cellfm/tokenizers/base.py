"""Tokenizer base interface and shared constants.

Shared special-token convention for gene-token encoders (rank, value_bin):
    PAD  = 0
    CLS  = 1
    MASK = 2
    real gene IDs start at N_SPECIAL = 3
    -> gene vocab size = n_genes + N_SPECIAL

For value_bin, value tokens use their own small vocab:
    PAD  = 0
    MASK = 1
    real bin IDs start at N_VALUE_SPECIAL = 2
    -> bin vocab size = n_bins + N_VALUE_SPECIAL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

PAD_ID: int = 0
CLS_ID: int = 1
MASK_ID: int = 2
N_SPECIAL: int = 3

VAL_PAD_ID: int = 0
VAL_MASK_ID: int = 1
N_VALUE_SPECIAL: int = 2


@dataclass
class TokenizerConfig:
    """Shared tokenizer configuration.

    Concrete tokenizers may ignore fields they do not need.
    """

    n_genes: int
    L: int = 2048
    hvg_indices: list[int] | None = None     # for HVG-dense
    n_bins: int = 51                         # for value_bin (locked: 51, scGPT)
    log1p_normalize: bool = True             # for HVG-dense + value_bin
    target_sum: float = 1e4                  # CPM-ish per-cell scaling for log1p
    mask_ratio: float = 0.15                 # masked-token ratio (rank / value_bin)
    add_cls: bool = True                     # prepend CLS for transformer encoders


class Tokenizer(Protocol):
    """Encoder interface. Implementations live in sibling modules."""

    cfg: TokenizerConfig

    def encode_batch(self, items: list[dict], *, train: bool = True) -> dict[str, torch.Tensor]:
        """Convert a list of CellShardDataset records into a batch dict."""
        ...

    @property
    def gene_vocab_size(self) -> int:
        ...

    @property
    def value_vocab_size(self) -> int | None:
        ...
