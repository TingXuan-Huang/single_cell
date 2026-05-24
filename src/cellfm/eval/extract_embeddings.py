"""Embedding extraction.

Run a trained model in eval mode over an entire split, return concatenated
embeddings + labels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
    max_batches: int | None = None,
) -> dict:
    """Return {'X': (N, d), 'y': (N,)} as numpy arrays."""
    device = device or next(model.parameters()).device
    model.eval()

    chunks_X: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        emb = model.extract_embedding(batch)
        chunks_X.append(emb.detach().cpu().numpy())
        if "labels" in batch:
            chunks_y.append(batch["labels"].detach().cpu().numpy())
    X = np.concatenate(chunks_X, axis=0) if chunks_X else np.zeros((0, 0))
    y = np.concatenate(chunks_y, axis=0) if chunks_y else np.zeros(0, dtype=np.int64)
    return {"X": X, "y": y}
