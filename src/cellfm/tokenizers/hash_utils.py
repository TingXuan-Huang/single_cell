"""Deterministic hash helpers for sparse gene encoders.

Python's built-in ``hash()`` is salted per process, so it is not suitable for
checkpoint-reproducible gene hashing. These helpers use fixed integer arithmetic
instead.
"""

from __future__ import annotations

import torch


def stable_multi_hash(
    gene_ids: torch.Tensor,
    *,
    n_buckets: int,
    n_hashes: int,
    salt: int = 0,
) -> torch.Tensor:
    """Map gene ids to ``n_hashes`` deterministic bucket ids.

    Args:
        gene_ids: Tensor of non-negative integer gene ids with any shape.
        n_buckets: Number of buckets per hash table.
        n_hashes: Number of independent hash tables.
        salt: Optional integer offset for reproducible alternate hash families.

    Returns:
        Long tensor with shape ``gene_ids.shape + (n_hashes,)``.
    """
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive")
    if n_hashes <= 0:
        raise ValueError("n_hashes must be positive")

    ids = gene_ids.to(dtype=torch.long).unsqueeze(-1)
    hash_i = torch.arange(n_hashes, dtype=torch.long, device=gene_ids.device)
    multipliers = 1_103_515_245 + 12_345 * (hash_i + 1)
    salts = 2_654_435_761 * (hash_i + 1) + int(salt)
    return torch.remainder(ids * multipliers + salts, int(n_buckets))
