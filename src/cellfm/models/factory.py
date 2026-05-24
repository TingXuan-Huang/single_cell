"""Model factory.

`build_model(encoder, size, n_genes, n_classes, L, n_hvg, ...)` returns a
ready-to-train nn.Module matching the requested encoder.

Size presets target the *transformer body* parameter count, not total params
(the gene embedding table dominates total params and is reported separately).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from cellfm.models.embedding_bag_mlp import EmbeddingBagConfig, EmbeddingBagMLP
from cellfm.models.hvg_mlp import HVGMLP, HVGMLPConfig
from cellfm.models.rank_transformer import RankTransformer, RankTransformerConfig
from cellfm.models.transformer_body import BodyConfig
from cellfm.models.value_bin_transformer import (
    ValueBinTransformer,
    ValueBinTransformerConfig,
)


@dataclass
class SizeConfig:
    """Size preset. For transformer encoders these set the body shape;
    for MLP baselines they set the trunk shape."""

    name: str
    d_model: int
    n_layers: int
    n_heads: int
    ffn_mult: int
    mlp_hidden: tuple[int, ...]
    mlp_d_embed: int


SIZE_CONFIGS: dict[str, SizeConfig] = {
    "tiny_1m": SizeConfig(
        name="tiny_1m",
        d_model=128,
        n_layers=2,
        n_heads=4,
        ffn_mult=4,
        mlp_hidden=(256, 256),
        mlp_d_embed=128,
    ),
    "tiny_3m": SizeConfig(
        name="tiny_3m",
        d_model=192,
        n_layers=4,
        n_heads=6,
        ffn_mult=4,
        mlp_hidden=(384, 384, 384),
        mlp_d_embed=192,
    ),
    # ~5M body params (rank/value_bin). MLP baselines size-matched on hidden trunk.
    "tiny_5m": SizeConfig(
        name="tiny_5m",
        d_model=256,
        n_layers=6,
        n_heads=8,
        ffn_mult=4,
        mlp_hidden=(512, 512, 512, 512),
        mlp_d_embed=256,
    ),
    # ~10M body params. head_dim=40 still hits SDPA fast path.
    "tiny_10m": SizeConfig(
        name="tiny_10m",
        d_model=320,
        n_layers=8,
        n_heads=8,
        ffn_mult=4,
        mlp_hidden=(640, 640, 640, 640, 640),
        mlp_d_embed=320,
    ),
}


def count_params(model: nn.Module, exclude: tuple[str, ...] = ()) -> dict[str, int]:
    """Count trainable params, with optional buckets for reporting.

    Params with a name containing any substring in `exclude` are skipped
    entirely (neither bucketed nor added to total). The default `()` counts
    everything and produces buckets ``{"embedding_table", "rest", "total"}``.
    """
    total = 0
    buckets: dict[str, int] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(ex in name for ex in exclude):
            continue
        n = p.numel()
        bucket = "embedding_table" if (
            "gene_emb" in name or "bag.weight" in name
        ) else "rest"
        buckets[bucket] = buckets.get(bucket, 0) + n
        total += n
    buckets["total"] = total
    return buckets


def build_model(
    *,
    encoder: str,
    size: str,
    n_genes: int,
    n_classes: int,
    L: int = 2048,
    n_hvg: int = 2000,
    gene_vocab_size: int | None = None,
    value_vocab_size: int | None = None,
    dropout: float = 0.1,
    mlm_weight: float = 1.0,
    ce_weight: float = 0.5,
) -> nn.Module:
    """Build a model from encoder name + size preset."""
    if size not in SIZE_CONFIGS:
        raise KeyError(f"Unknown size '{size}'. Available: {list(SIZE_CONFIGS)}")
    sz = SIZE_CONFIGS[size]

    if encoder == "hvg_dense":
        cfg = HVGMLPConfig(
            n_hvg=n_hvg,
            hidden_dims=sz.mlp_hidden,
            dropout=dropout,
            n_classes=n_classes,
        )
        return HVGMLP(cfg)
    if encoder == "embedding_bag":
        cfg = EmbeddingBagConfig(
            n_genes=n_genes,
            d_embed=sz.mlp_d_embed,
            hidden_dims=sz.mlp_hidden,
            dropout=dropout,
            n_classes=n_classes,
            mlm_weight=mlm_weight,
            ce_weight=ce_weight,
        )
        return EmbeddingBagMLP(cfg)

    body = BodyConfig(
        d_model=sz.d_model,
        n_layers=sz.n_layers,
        n_heads=sz.n_heads,
        ffn_mult=sz.ffn_mult,
        dropout=dropout,
        max_len=L,
    )
    if encoder == "rank":
        if gene_vocab_size is None:
            raise ValueError("rank encoder needs gene_vocab_size")
        cfg = RankTransformerConfig(
            gene_vocab_size=gene_vocab_size,
            L=L,
            body=body,
            n_classes=n_classes,
            mlm_weight=mlm_weight,
            ce_weight=ce_weight,
        )
        return RankTransformer(cfg)
    if encoder == "value_bin":
        if gene_vocab_size is None or value_vocab_size is None:
            raise ValueError("value_bin encoder needs gene_vocab_size and value_vocab_size")
        cfg = ValueBinTransformerConfig(
            gene_vocab_size=gene_vocab_size,
            value_vocab_size=value_vocab_size,
            L=L,
            body=body,
            n_classes=n_classes,
            mlm_gene_weight=mlm_weight,
            mlm_value_weight=mlm_weight,
            ce_weight=ce_weight,
        )
        return ValueBinTransformer(cfg)
    raise KeyError(
        f"Unknown encoder '{encoder}'. Available: hvg_dense, embedding_bag, rank, value_bin"
    )
