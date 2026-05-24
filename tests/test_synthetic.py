"""Sanity checks on the synthetic AnnData generator."""

from __future__ import annotations

from scipy import sparse


def test_synthetic_shape_and_sparsity(synthetic_adata):
    assert synthetic_adata.n_obs == 400
    assert synthetic_adata.n_vars == 300
    assert sparse.issparse(synthetic_adata.X)
    # Reasonable sparsity
    nnz = synthetic_adata.X.nnz
    total = synthetic_adata.X.shape[0] * synthetic_adata.X.shape[1]
    density = nnz / total
    assert 0.02 < density < 0.20


def test_synthetic_metadata(synthetic_adata):
    obs = synthetic_adata.obs
    var = synthetic_adata.var
    assert "subclass" in obs.columns
    assert "donor_id" in obs.columns
    assert "pct_mt" in obs.columns
    assert "mt" in var.columns
    assert obs["subclass"].nunique() <= 6
    assert obs["donor_id"].nunique() <= 8
