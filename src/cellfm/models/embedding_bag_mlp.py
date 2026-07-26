"""Weighted-EmbeddingBag baseline with supervised + masked-reconstruction objectives.

This is mathematically a sparse linear layer (see memory/LEARNINGS.md
#sparse-linear-equals-weighted-bag) followed by a small MLP. We expose both:

- Supervised path: classify subclass from the bag embedding.
- Self-supervised path: reconstruct masked gene values from the bag embedding,
  which forces the embedding to retain within-cell expression info.

This means EmbeddingBag, unlike HVG-dense, has *both* an objective family,
mirroring how the transformer encoders combine supervised + MLM heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EmbeddingBagConfig:
    n_genes: int
    d_embed: int = 128
    hidden_dims: tuple[int, ...] = (256, 256)
    dropout: float = 0.1
    n_classes: int = 0
    mlm_weight: float = 1.0
    ce_weight: float = 1.0


# fusion: model = EmbeddingBagMLP(EmbeddingBagConfig(n_genes=1000, n_classes=10))
class EmbeddingBagMLP(nn.Module):
    name = "embedding_bag"

    def __init__(self, cfg: EmbeddingBagConfig):
        super().__init__()
        self.cfg = cfg
        # mode='sum' with per_sample_weights replicates sparse linear: x @ W.T.
        # sparse=False because optimizer choice (AdamW) doesn't support sparse grads.
        self.bag = nn.EmbeddingBag(cfg.n_genes, cfg.d_embed, mode="sum", sparse=False)

        dims = [cfg.d_embed, *cfg.hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(cfg.dropout))
        self.trunk = nn.Sequential(*layers)
        self.classifier = nn.Linear(cfg.hidden_dims[-1], cfg.n_classes)
        # Reconstruction: predict a scalar value for each masked (cell, gene).
        # Use a tied bilinear: pred = <h_cell, W_gene^reconstruct[gene]>
        self.value_proj = nn.Linear(cfg.hidden_dims[-1], cfg.d_embed)

    @property
    def d_embedding(self) -> int:
        return self.cfg.hidden_dims[-1]
# fusion: input = {"indices": torch.randint(0, 1000, (100,)), "offsets": torch.arange(0, 100, 10), "per_sample_weights": torch.randn(100)}

    def _bag_embed(self, batch: dict) -> torch.Tensor:
        return self.bag(
            input=batch["indices"],
            offsets=batch["offsets"],
            per_sample_weights=batch["per_sample_weights"],
        # fusion: input = {"indices": torch.randint(0, 1000, (10,)), "offsets": torch.tensor([0, 5], dtype=torch.long), "per_sample_weights": torch.randn(10)}
        )

    def extract_embedding(self, batch: dict) -> torch.Tensor:
        # fusion: input = {"indices": torch.randint(0, 1000, (32,)), "offsets": torch.tensor([0, 4, 8, 12, 16, 20, 24, 28]), "per_sample_weights": torch.randn(32), "labels": torch.randint(0, 10, (8,)), "masked_cells": torch.randint(0, 8, (5,)), "masked_genes": torch.randint(0, 1000, (5,)), "masked_values": torch.randn(5)}
        z = self._bag_embed(batch)
        return self.trunk(z)

    def forward(self, batch: dict) -> dict:
        z = self._bag_embed(batch)
        h = self.trunk(z)
        logits = self.classifier(h)
        out: dict = {"logits": logits, "embedding": h}

        if "labels" in batch:
            ce = F.cross_entropy(logits, batch["labels"], ignore_index=-100)
            out["ce_loss"] = ce

        recon_loss = torch.tensor(0.0, device=z.device)
        if "masked_cells" in batch and batch["masked_cells"].numel() > 0:
            mc = batch["masked_cells"]
            mg = batch["masked_genes"]
            tgt = batch["masked_values"]
            # gene embedding for masked gene IDs
            gene_emb = self.bag.weight[mg]                # (M, d_embed)
            cell_h = self.value_proj(h[mc])               # (M, d_embed)
            preds = (cell_h * gene_emb).sum(dim=-1)       # (M,)
            recon_loss = F.mse_loss(preds, tgt)
            out["recon_loss"] = recon_loss

        total = self.cfg.ce_weight * out.get(
            "ce_loss", torch.tensor(0.0, device=z.device)
        ) + self.cfg.mlm_weight * recon_loss
        out["loss"] = total
        return out
