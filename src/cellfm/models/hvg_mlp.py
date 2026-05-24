"""HVG-dense MLP supervised baseline.

Input: (B, n_hvg) float32 of log1p-normalized expression.
Output: (B, n_classes) logits.
Embedding for downstream evaluation: penultimate hidden layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HVGMLPConfig:
    n_hvg: int
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    dropout: float = 0.1
    n_classes: int = 0


class HVGMLP(nn.Module):
    name = "hvg_dense"

    def __init__(self, cfg: HVGMLPConfig):
        super().__init__()
        self.cfg = cfg
        dims = [cfg.n_hvg, *cfg.hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(cfg.dropout))
        self.trunk = nn.Sequential(*layers)
        self.classifier = nn.Linear(cfg.hidden_dims[-1], cfg.n_classes)

    @property
    def d_embedding(self) -> int:
        return self.cfg.hidden_dims[-1]

    def extract_embedding(self, batch: dict) -> torch.Tensor:
        x = batch["x_dense"]
        return self.trunk(x)

    def forward(self, batch: dict) -> dict:
        x = batch["x_dense"]
        h = self.trunk(x)
        logits = self.classifier(h)
        out: dict = {"logits": logits, "embedding": h}
        if "labels" in batch:
            out["ce_loss"] = F.cross_entropy(logits, batch["labels"], ignore_index=-100)
            out["loss"] = out["ce_loss"]
        return out
