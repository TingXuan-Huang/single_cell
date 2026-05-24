"""Tokenizer behavior tests against the synthetic cache."""

from __future__ import annotations

import numpy as np
import torch

from cellfm.data.cache import CacheManifest, CellShardDataset
from cellfm.tokenizers.base import N_SPECIAL, PAD_ID, TokenizerConfig
from cellfm.tokenizers.embedding_bag import EmbeddingBagTokenizer
from cellfm.tokenizers.hvg_dense import HVGDenseTokenizer
from cellfm.tokenizers.rank import RankTokenizer
from cellfm.tokenizers.value_bin import ValueBinTokenizer
from cellfm.training import build_dataloaders


def _items(cache_dir, k=8):
    ds = CellShardDataset(cache_dir, split="train")
    return [ds[i] for i in range(min(k, len(ds)))], ds.manifest


def test_hvg_dense_tokenizer(synthetic_cache):
    items, manifest = _items(synthetic_cache)
    cfg = TokenizerConfig(n_genes=manifest.n_genes, L=manifest.L,
                          hvg_indices=manifest.hvg_indices)
    tok = HVGDenseTokenizer(cfg)
    batch = tok.encode_batch(items, train=True)
    assert batch["x_dense"].shape == (len(items), len(manifest.hvg_indices))
    assert batch["labels"].shape == (len(items),)
    assert batch["x_dense"].dtype == torch.float32
    # log1p output is non-negative
    assert (batch["x_dense"] >= 0).all()


def test_embedding_bag_tokenizer(synthetic_cache):
    items, manifest = _items(synthetic_cache)
    cfg = TokenizerConfig(n_genes=manifest.n_genes, L=manifest.L, mask_ratio=0.15)
    tok = EmbeddingBagTokenizer(cfg)
    b = tok.encode_batch(items, train=True)
    assert b["offsets"].shape == (len(items),)
    assert b["indices"].dim() == 1 and b["per_sample_weights"].dim() == 1
    assert b["indices"].numel() == b["per_sample_weights"].numel()
    # Train mode -> some masked entries
    assert b["masked_genes"].numel() > 0


def test_rank_tokenizer_special_tokens(synthetic_cache):
    items, manifest = _items(synthetic_cache)
    cfg = TokenizerConfig(n_genes=manifest.n_genes, L=manifest.L, mask_ratio=0.15,
                          add_cls=True)
    tok = RankTokenizer(cfg)
    b = tok.encode_batch(items, train=True)
    assert b["input_ids"].shape == (len(items), manifest.L)
    # First position is CLS
    assert (b["input_ids"][:, 0] == 1).all()
    # Padding mask aligns
    assert b["attention_mask"].dtype == torch.bool
    # MLM targets are -100 except at masked positions
    tgt = b["mlm_targets"].numpy()
    n_masked = (tgt != -100).sum()
    assert n_masked > 0


def test_value_bin_tokenizer_bins(synthetic_cache):
    items, manifest = _items(synthetic_cache)
    cfg = TokenizerConfig(n_genes=manifest.n_genes, L=manifest.L, n_bins=51,
                          mask_ratio=0.15, add_cls=True)
    tok = ValueBinTokenizer(cfg)
    b = tok.encode_batch(items, train=True)
    assert b["gene_ids"].shape == (len(items), manifest.L)
    assert b["value_ids"].shape == (len(items), manifest.L)
    # Value IDs should fit within vocab
    assert tok.value_vocab_size is not None
    assert int(b["value_ids"].max()) < tok.value_vocab_size


def test_eval_mode_disables_masking(synthetic_cache):
    items, manifest = _items(synthetic_cache)
    cfg = TokenizerConfig(n_genes=manifest.n_genes, L=manifest.L, mask_ratio=0.5,
                          add_cls=True)
    tok = RankTokenizer(cfg)
    b_train = tok.encode_batch(items, train=True)
    b_eval = tok.encode_batch(items, train=False)
    assert (b_eval["mlm_targets"].numpy() == -100).all()
    # train mode produces real targets
    assert (b_train["mlm_targets"].numpy() != -100).any()


def test_training_validation_loader_can_mask_eval_splits(synthetic_cache):
    items, manifest = _items(synthetic_cache)
    cfg = TokenizerConfig(
        n_genes=manifest.n_genes,
        L=manifest.L,
        n_bins=51,
        mask_ratio=0.5,
        add_cls=True,
    )
    tok = ValueBinTokenizer(cfg)
    loaders = build_dataloaders(
        synthetic_cache,
        tok,
        batch_size=len(items),
        num_workers=0,
        mask_eval_splits=True,
    )
    val_batch = next(iter(loaders["val"]))
    assert (val_batch["mlm_gene_targets"].numpy() != -100).any()
    assert (val_batch["mlm_value_targets"].numpy() != -100).any()

    eval_loaders = build_dataloaders(
        synthetic_cache,
        tok,
        batch_size=len(items),
        num_workers=0,
        eval_mode=True,
        mask_eval_splits=True,
    )
    eval_val_batch = next(iter(eval_loaders["val"]))
    assert (eval_val_batch["mlm_gene_targets"].numpy() == -100).all()
    assert (eval_val_batch["mlm_value_targets"].numpy() == -100).all()
