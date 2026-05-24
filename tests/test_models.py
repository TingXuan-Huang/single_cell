"""Smoke tests: each model + tokenizer combination produces a backward-compatible loss."""

from __future__ import annotations

import torch

from cellfm.data.cache import CellShardDataset
from cellfm.models import SIZE_CONFIGS, build_model, count_params
from cellfm.tokenizers.base import TokenizerConfig
from cellfm.tokenizers.embedding_bag import EmbeddingBagTokenizer
from cellfm.tokenizers.hvg_dense import HVGDenseTokenizer
from cellfm.tokenizers.rank import RankTokenizer
from cellfm.tokenizers.value_bin import ValueBinTokenizer


def _batch(cache_dir, tokenizer, k=4):
    ds = CellShardDataset(cache_dir, split="train")
    items = [ds[i] for i in range(min(k, len(ds)))]
    return tokenizer.encode_batch(items, train=True), ds.manifest


def test_hvg_mlp_forward_backward(synthetic_cache):
    from cellfm.data.cache import CacheManifest
    m = CacheManifest.from_json(synthetic_cache / "manifest.json")
    tok = HVGDenseTokenizer(TokenizerConfig(n_genes=m.n_genes, L=m.L,
                                            hvg_indices=m.hvg_indices))
    batch, _ = _batch(synthetic_cache, tok)
    model = build_model(encoder="hvg_dense", size="tiny_1m",
                        n_genes=m.n_genes, n_classes=len(m.label_vocab),
                        L=m.L, n_hvg=len(m.hvg_indices))
    out = model(batch)
    assert "loss" in out and out["loss"].requires_grad
    out["loss"].backward()


def test_embedding_bag_forward_backward(synthetic_cache):
    from cellfm.data.cache import CacheManifest
    m = CacheManifest.from_json(synthetic_cache / "manifest.json")
    tok = EmbeddingBagTokenizer(TokenizerConfig(n_genes=m.n_genes, L=m.L,
                                                mask_ratio=0.15))
    batch, _ = _batch(synthetic_cache, tok)
    model = build_model(encoder="embedding_bag", size="tiny_1m",
                        n_genes=m.n_genes, n_classes=len(m.label_vocab))
    out = model(batch)
    assert "loss" in out
    out["loss"].backward()


def test_rank_transformer_forward_backward(synthetic_cache):
    from cellfm.data.cache import CacheManifest
    m = CacheManifest.from_json(synthetic_cache / "manifest.json")
    tok = RankTokenizer(TokenizerConfig(n_genes=m.n_genes, L=m.L,
                                        mask_ratio=0.15, add_cls=True))
    batch, _ = _batch(synthetic_cache, tok)
    model = build_model(encoder="rank", size="tiny_1m",
                        n_genes=m.n_genes, n_classes=len(m.label_vocab),
                        L=m.L, gene_vocab_size=tok.gene_vocab_size)
    out = model(batch)
    assert "loss" in out
    out["loss"].backward()


def test_rank_transformer_mlm_logits_only_masked_positions(synthetic_cache):
    from cellfm.data.cache import CacheManifest
    m = CacheManifest.from_json(synthetic_cache / "manifest.json")
    tok = RankTokenizer(TokenizerConfig(n_genes=m.n_genes, L=m.L,
                                        mask_ratio=0.15, add_cls=True))
    batch, _ = _batch(synthetic_cache, tok)
    model = build_model(encoder="rank", size="tiny_1m",
                        n_genes=m.n_genes, n_classes=len(m.label_vocab),
                        L=m.L, gene_vocab_size=tok.gene_vocab_size)
    orig = model._mlm_logits
    seen = {}

    def wrapped(h):
        seen["shape"] = tuple(h.shape)
        return orig(h)

    model._mlm_logits = wrapped
    out = model(batch)
    assert "loss" in out
    assert len(seen["shape"]) == 2
    assert seen["shape"][0] == int((batch["mlm_targets"] != -100).sum())


def test_value_bin_transformer_forward_backward(synthetic_cache):
    from cellfm.data.cache import CacheManifest
    m = CacheManifest.from_json(synthetic_cache / "manifest.json")
    tok = ValueBinTokenizer(TokenizerConfig(n_genes=m.n_genes, L=m.L,
                                            mask_ratio=0.15, add_cls=True))
    batch, _ = _batch(synthetic_cache, tok)
    model = build_model(encoder="value_bin", size="tiny_1m",
                        n_genes=m.n_genes, n_classes=len(m.label_vocab),
                        L=m.L,
                        gene_vocab_size=tok.gene_vocab_size,
                        value_vocab_size=tok.value_vocab_size)
    out = model(batch)
    assert "loss" in out
    out["loss"].backward()


def test_value_bin_transformer_gene_logits_only_masked_positions(synthetic_cache):
    from cellfm.data.cache import CacheManifest
    m = CacheManifest.from_json(synthetic_cache / "manifest.json")
    tok = ValueBinTokenizer(TokenizerConfig(n_genes=m.n_genes, L=m.L,
                                            mask_ratio=0.15, add_cls=True))
    batch, _ = _batch(synthetic_cache, tok)
    model = build_model(encoder="value_bin", size="tiny_1m",
                        n_genes=m.n_genes, n_classes=len(m.label_vocab),
                        L=m.L,
                        gene_vocab_size=tok.gene_vocab_size,
                        value_vocab_size=tok.value_vocab_size)
    orig = model._gene_logits
    seen = {}

    def wrapped(h):
        seen["shape"] = tuple(h.shape)
        return orig(h)

    model._gene_logits = wrapped
    out = model(batch)
    assert "loss" in out
    assert len(seen["shape"]) == 2
    assert seen["shape"][0] == int((batch["mlm_gene_targets"] != -100).sum())


def test_count_params_separates_embedding_table(synthetic_cache):
    from cellfm.data.cache import CacheManifest
    m = CacheManifest.from_json(synthetic_cache / "manifest.json")
    tok = RankTokenizer(TokenizerConfig(n_genes=m.n_genes, L=m.L,
                                        mask_ratio=0.15, add_cls=True))
    model = build_model(encoder="rank", size="tiny_1m",
                        n_genes=m.n_genes, n_classes=len(m.label_vocab),
                        L=m.L, gene_vocab_size=tok.gene_vocab_size)
    counts = count_params(model)
    assert "embedding_table" in counts
    assert "rest" in counts
    assert counts["embedding_table"] > 0
    assert counts["rest"] > 0


def test_extract_embedding_shape(synthetic_cache):
    from cellfm.data.cache import CacheManifest
    m = CacheManifest.from_json(synthetic_cache / "manifest.json")
    tok = RankTokenizer(TokenizerConfig(n_genes=m.n_genes, L=m.L,
                                        mask_ratio=0.0, add_cls=True))
    batch, _ = _batch(synthetic_cache, tok)
    model = build_model(encoder="rank", size="tiny_1m",
                        n_genes=m.n_genes, n_classes=len(m.label_vocab),
                        L=m.L, gene_vocab_size=tok.gene_vocab_size)
    model.eval()
    with torch.no_grad():
        emb = model.extract_embedding(batch)
    assert emb.shape == (batch["input_ids"].shape[0], SIZE_CONFIGS["tiny_1m"].d_model)
