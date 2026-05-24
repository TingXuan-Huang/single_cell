"""Rank-token transformer cell FM.

Architecture:
  gene_ids (B, L) int -> gene_embedding (B, L, d) + pos_embedding -> transformer ->
  -> CLS pooling -> cell embedding (B, d)
  -> classifier (supervised side-objective, not load-bearing)
  -> MLM head over gene_vocab (load-bearing pretraining objective)

The classifier is included for fair comparison with baselines, but the
load-bearing training signal is masked gene ID prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellfm.models.transformer_body import BodyConfig, TransformerEncoder


@dataclass
class RankTransformerConfig:
    gene_vocab_size: int       # = n_genes + N_SPECIAL
    L: int = 2048
    body: BodyConfig = None
    n_classes: int = 0
    mlm_weight: float = 1.0
    ce_weight: float = 0.5
    pad_id: int = 0


class RankTransformer(nn.Module):
    name = "rank"

    def __init__(self, cfg: RankTransformerConfig):
        super().__init__()
        if cfg.body is None:
            cfg.body = BodyConfig(max_len=cfg.L)
        self.cfg = cfg
        d = cfg.body.d_model
        self.gene_emb = nn.Embedding(cfg.gene_vocab_size, d, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.L, d)
        self.encoder = TransformerEncoder(cfg.body)
        self.classifier = nn.Linear(d, cfg.n_classes)
        # MLM head: project to gene vocab. Weight-tied to gene_emb for parameter savings.
        self.mlm_proj = nn.Linear(d, d)
        # Optional: bias for tied softmax
        self.mlm_bias = nn.Parameter(torch.zeros(cfg.gene_vocab_size))

    @property
    def d_embedding(self) -> int:
        return self.cfg.body.d_model

    def _embed(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.gene_emb(ids) + self.pos_emb(pos)
        h = self.encoder(x, attn_mask=mask)
        return h, mask

    def _mlm_logits(self, h: torch.Tensor) -> torch.Tensor:
        # tied softmax: logits = (W h)  W = gene_emb.weight (V, d) -> (B, L, V)
        h_proj = self.mlm_proj(h)
        return h_proj @ self.gene_emb.weight.t() + self.mlm_bias

    def extract_embedding(self, batch: dict) -> torch.Tensor:
        h, _mask = self._embed(batch)
        return h[:, 0, :]   # CLS token at position 0

    def forward(self, batch: dict) -> dict:
        h, _mask = self._embed(batch)
        cls = h[:, 0, :]
        logits = self.classifier(cls)
        out: dict = {"logits": logits, "embedding": cls}

        ce_loss = torch.tensor(0.0, device=h.device)
        if "labels" in batch:
            ce_loss = F.cross_entropy(logits, batch["labels"], ignore_index=-100)
            out["ce_loss"] = ce_loss

        mlm_loss = torch.tensor(0.0, device=h.device)
        if "mlm_targets" in batch:
            mlm_targets = batch["mlm_targets"]
            masked = mlm_targets != -100
            if masked.any():
                # Full (B, L, V) logits are too large for Allen-scale vocabularies.
                # Compute the tied softmax only at masked positions.
                mlm_logits = self._mlm_logits(h[masked])
                mlm_loss = F.cross_entropy(mlm_logits, mlm_targets[masked])
            out["mlm_loss"] = mlm_loss

        out["loss"] = self.cfg.ce_weight * ce_loss + self.cfg.mlm_weight * mlm_loss
        return out
