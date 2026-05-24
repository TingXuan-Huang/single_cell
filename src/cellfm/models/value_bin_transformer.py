"""Value-bin (scGPT-style) transformer cell FM.

Architecture:
  gene_ids + value_bin_ids -> gene_emb + value_emb -> sum -> + pos_emb ->
  transformer -> CLS pooling -> cell embedding
  -> classifier (supervised side-objective)
  -> joint MLM head: predict masked gene id AND masked value bin

The joint objective is the load-bearing pretraining signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellfm.models.transformer_body import BodyConfig, TransformerEncoder


@dataclass
class ValueBinTransformerConfig:
    gene_vocab_size: int        # n_genes + N_SPECIAL
    value_vocab_size: int       # n_bins + N_VALUE_SPECIAL
    L: int = 2048
    body: BodyConfig = None
    n_classes: int = 0
    mlm_gene_weight: float = 1.0
    mlm_value_weight: float = 1.0
    ce_weight: float = 0.5
    pad_id: int = 0
    val_pad_id: int = 0


class ValueBinTransformer(nn.Module):
    name = "value_bin"

    def __init__(self, cfg: ValueBinTransformerConfig):
        super().__init__()
        if cfg.body is None:
            cfg.body = BodyConfig(max_len=cfg.L)
        self.cfg = cfg
        d = cfg.body.d_model
        self.gene_emb = nn.Embedding(cfg.gene_vocab_size, d, padding_idx=cfg.pad_id)
        self.value_emb = nn.Embedding(cfg.value_vocab_size, d, padding_idx=cfg.val_pad_id)
        self.pos_emb = nn.Embedding(cfg.L, d)
        self.encoder = TransformerEncoder(cfg.body)
        self.classifier = nn.Linear(d, cfg.n_classes)
        self.mlm_gene_proj = nn.Linear(d, d)
        self.mlm_gene_bias = nn.Parameter(torch.zeros(cfg.gene_vocab_size))
        self.mlm_value_head = nn.Linear(d, cfg.value_vocab_size)

    @property
    def d_embedding(self) -> int:
        return self.cfg.body.d_model

    def _embed(self, batch: dict) -> torch.Tensor:
        gid = batch["gene_ids"]
        vid = batch["value_ids"]
        mask = batch["attention_mask"]
        B, L = gid.shape
        pos = torch.arange(L, device=gid.device).unsqueeze(0).expand(B, L)
        x = self.gene_emb(gid) + self.value_emb(vid) + self.pos_emb(pos)
        h = self.encoder(x, attn_mask=mask)
        return h

    def _gene_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlm_gene_proj(h) @ self.gene_emb.weight.t() + self.mlm_gene_bias

    def extract_embedding(self, batch: dict) -> torch.Tensor:
        h = self._embed(batch)
        return h[:, 0, :]

    def forward(self, batch: dict) -> dict:
        h = self._embed(batch)
        cls = h[:, 0, :]
        logits = self.classifier(cls)
        out: dict = {"logits": logits, "embedding": cls}

        ce_loss = torch.tensor(0.0, device=h.device)
        if "labels" in batch:
            ce_loss = F.cross_entropy(logits, batch["labels"], ignore_index=-100)
            out["ce_loss"] = ce_loss

        mlm_gene_loss = torch.tensor(0.0, device=h.device)
        mlm_val_loss = torch.tensor(0.0, device=h.device)

        if "mlm_gene_targets" in batch:
            g_logits = self._gene_logits(h)
            B, L, V = g_logits.shape
            mlm_gene_loss = F.cross_entropy(
                g_logits.reshape(B * L, V),
                batch["mlm_gene_targets"].reshape(B * L),
                ignore_index=-100,
            )
            out["mlm_gene_loss"] = mlm_gene_loss

        if "mlm_value_targets" in batch:
            v_logits = self.mlm_value_head(h)
            B, L, Vb = v_logits.shape
            mlm_val_loss = F.cross_entropy(
                v_logits.reshape(B * L, Vb),
                batch["mlm_value_targets"].reshape(B * L),
                ignore_index=-100,
            )
            out["mlm_value_loss"] = mlm_val_loss

        out["loss"] = (
            self.cfg.ce_weight * ce_loss
            + self.cfg.mlm_gene_weight * mlm_gene_loss
            + self.cfg.mlm_value_weight * mlm_val_loss
        )
        return out
