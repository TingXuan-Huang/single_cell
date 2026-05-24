"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_adata():
    from cellfm.data.synthetic import make_synthetic_anndata

    return make_synthetic_anndata(
        n_cells=400,
        n_genes=300,
        n_subclasses=6,
        n_donors=8,
        sparsity=0.92,
        seed=0,
    )


@pytest.fixture(scope="session")
def synthetic_cache(synthetic_adata, tmp_path_factory) -> Path:
    """Build a real on-disk cache from the synthetic AnnData for downstream tests."""
    from cellfm.data.cache import write_cache
    from cellfm.data.hvg import select_hvg
    from cellfm.data.qc import QCConfig, compute_nnz_distribution, pick_L_power_of_two, run_qc
    from cellfm.data.splits import SplitConfig, donor_stratified_split

    adata = synthetic_adata.copy()
    adata, _qc = run_qc(adata, QCConfig(min_nnz=10, min_total_counts=30, max_pct_mt=100.0,
                                        max_doublet_score=None))
    sp = SplitConfig(donor_col="donor_id", label_col="subclass",
                     train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=0,
                     min_donors_per_split=1)
    indices = donor_stratified_split(adata, sp)
    train_adata = adata[indices["train"]]
    hvg = select_hvg(train_adata, n_top=64, flavor="seurat_v3")
    nnz = compute_nnz_distribution(train_adata)
    L = max(64, min(256, pick_L_power_of_two(int(nnz["p95"]))))

    out = tmp_path_factory.mktemp("cache")
    write_cache(adata, indices, hvg, L, "subclass", out,
                qc_summary={"nnz_stats_train": nnz})
    return out


@pytest.fixture
def rng_seed_zero():
    return np.random.default_rng(0)
