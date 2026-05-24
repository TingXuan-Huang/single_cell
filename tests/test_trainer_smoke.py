"""End-to-end smoke test: build dataloaders + train for a handful of steps.

Parametrized across all 4 (encoder, size) combinations used in the v1 sweep so
that *every* config path in `configs/model/*.yaml` is exercised before launching
a SLURM array. The synthetic cache has L ≤ 64 and the batch size here is 4, so
the test stays CPU-runnable in a few seconds total.
"""

from __future__ import annotations

import pytest

from cellfm.data.cache import CacheManifest
from cellfm.models import build_model
from cellfm.tokenizers.base import TokenizerConfig
from cellfm.tokenizers.embedding_bag import EmbeddingBagTokenizer
from cellfm.tokenizers.hvg_dense import HVGDenseTokenizer
from cellfm.tokenizers.rank import RankTokenizer
from cellfm.tokenizers.value_bin import ValueBinTokenizer
from cellfm.training import Trainer, build_dataloaders
from cellfm.training.loop import TrainConfig


def _build_tokenizer(encoder: str, m: CacheManifest):
    """Mirror scripts.train_one tokenizer construction for each encoder."""
    base = TokenizerConfig(
        n_genes=m.n_genes,
        L=m.L,
        hvg_indices=m.hvg_indices,
        n_bins=51,
        mask_ratio=0.15,
        add_cls=True,
    )
    if encoder == "hvg_dense":
        return HVGDenseTokenizer(base)
    if encoder == "embedding_bag":
        return EmbeddingBagTokenizer(base)
    if encoder == "rank":
        return RankTokenizer(base)
    if encoder == "value_bin":
        return ValueBinTokenizer(base)
    raise KeyError(encoder)


def _build_kwargs(encoder: str, tok, m: CacheManifest) -> dict:
    """Encoder-specific build_model kwargs (factory needs different extras per encoder)."""
    extras: dict = {}
    if encoder == "rank":
        extras["gene_vocab_size"] = tok.gene_vocab_size
    if encoder == "value_bin":
        extras["gene_vocab_size"] = tok.gene_vocab_size
        extras["value_vocab_size"] = tok.value_vocab_size
    return extras


# 4 encoders x 4 sizes = 16 combos. Each parametrize id runs ~1-2 s on CPU
# because the synthetic cache is tiny (L=64, 4-cell batch, 6 train steps).
ENCODERS = ["hvg_dense", "embedding_bag", "rank", "value_bin"]
SIZES = ["tiny_1m", "tiny_3m", "tiny_5m", "tiny_10m"]


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("size", SIZES)
def test_trainer_smoke_all_sizes(encoder, size, synthetic_cache, tmp_path):
    m = CacheManifest.from_json(synthetic_cache / "manifest.json")
    tok = _build_tokenizer(encoder, m)
    loaders = build_dataloaders(
        cache_dir=synthetic_cache,
        tokenizer=tok,
        batch_size=4,
        num_workers=0,
        drop_last=True,
    )
    model = build_model(
        encoder=encoder, size=size,
        n_genes=m.n_genes, n_classes=len(m.label_vocab),
        L=m.L, n_hvg=len(m.hvg_indices),
        **_build_kwargs(encoder, tok, m),
    )
    cfg = TrainConfig(
        out_dir=tmp_path / f"smoke_{encoder}_{size}",
        encoder=encoder, size=size,
        n_steps=4, eval_every=2, warmup_steps=1, lr=1e-3, amp=False,
        log_every=2,
    )
    trainer = Trainer(model=model, loaders=loaders, cfg=cfg)
    trainer.fit()
    assert (cfg.out_dir / "final.pt").exists()
    assert (cfg.out_dir / "train_history.json").exists()
