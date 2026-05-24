"""Synthetic AnnData generator for unit tests.

Used by tests/ to avoid depending on real Allen data. Models a tiny multi-donor,
multi-subclass scRNA-seq dataset with sparse Poisson counts.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def make_synthetic_anndata(
    n_cells: int = 1000,
    n_genes: int = 500,
    n_subclasses: int = 8,
    n_donors: int = 5,
    sparsity: float = 0.92,
    seed: int = 0,
) -> ad.AnnData:
    """Generate a synthetic AnnData mimicking Allen WMB-10X structure.

    Args:
        n_cells: total cells
        n_genes: total genes
        n_subclasses: number of cell subclasses (labels for classification)
        n_donors: number of donors (for donor-stratified split testing)
        sparsity: fraction of zero entries (Allen WMB-10X is ~0.93)
        seed: RNG seed

    Returns:
        AnnData with .X (sparse CSR int counts), .obs (cell_id, subclass, class, donor_id),
        .var (gene_symbol, ensembl_id, highly_variable_proxy).
    """
    rng = np.random.default_rng(seed)

    # Subclass labels: each subclass has a 'signature' set of ~5% of genes that are upregulated.
    signature_size = max(5, n_genes // 20)
    signatures = {
        sc: rng.choice(n_genes, size=signature_size, replace=False)
        for sc in range(n_subclasses)
    }

    # Cell-type assignment (imbalanced to mimic real data: a power-law-ish distribution)
    class_weights = np.power(1.0 / (np.arange(n_subclasses) + 1), 1.5)
    class_weights /= class_weights.sum()
    cell_subclass = rng.choice(n_subclasses, size=n_cells, p=class_weights)

    # Donor assignment: each donor mostly has one subset of subclasses,
    # but every subclass appears in >=2 donors so donor-stratified splits work.
    cell_donor = rng.integers(0, n_donors, size=n_cells)

    # Build sparse count matrix.
    # Approach: per-cell, draw nnz from a distribution targeting the sparsity,
    # then sample genes biased toward the subclass signature.
    target_nnz_per_cell = max(10, int(n_genes * (1 - sparsity)))

    rows: list[int] = []
    cols: list[int] = []
    vals: list[int] = []

    for c in range(n_cells):
        nnz_c = max(5, int(rng.normal(target_nnz_per_cell, target_nnz_per_cell * 0.15)))
        nnz_c = min(nnz_c, n_genes)

        sig = signatures[cell_subclass[c]]
        # 60% from signature, 40% random
        n_sig = min(len(sig), int(nnz_c * 0.6))
        n_bg = nnz_c - n_sig

        gene_idx_sig = rng.choice(sig, size=n_sig, replace=False)
        bg_pool = np.setdiff1d(np.arange(n_genes), gene_idx_sig, assume_unique=False)
        gene_idx_bg = rng.choice(bg_pool, size=n_bg, replace=False)
        gene_idx = np.concatenate([gene_idx_sig, gene_idx_bg])

        # Counts: Poisson, signature genes have higher rate
        rate_sig = 8.0
        rate_bg = 1.5
        counts_sig = rng.poisson(rate_sig, size=n_sig).astype(np.int32)
        counts_bg = rng.poisson(rate_bg, size=n_bg).astype(np.int32)
        counts = np.concatenate([counts_sig, counts_bg])

        # Drop zeros that snuck in from Poisson
        mask = counts > 0
        gene_idx = gene_idx[mask]
        counts = counts[mask]

        rows.extend([c] * len(gene_idx))
        cols.extend(gene_idx.tolist())
        vals.extend(counts.tolist())

    X = sparse.csr_matrix(
        (np.asarray(vals, dtype=np.int32), (rows, cols)),
        shape=(n_cells, n_genes),
    )

    # Metadata
    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i:06d}" for i in range(n_cells)],
            "subclass": [f"sub_{s}" for s in cell_subclass],
            "class": [f"class_{s // 2}" for s in cell_subclass],
            "donor_id": [f"donor_{d}" for d in cell_donor],
            "library_id": [f"lib_{cell_donor[i]}_{rng.integers(0, 3)}" for i in range(n_cells)],
            "pct_mt": rng.uniform(0.5, 12.0, size=n_cells).astype(np.float32),
            "doublet_score": rng.uniform(0.0, 0.4, size=n_cells).astype(np.float32),
        }
    )
    obs.index = obs["cell_id"].values

    var = pd.DataFrame(
        {
            "gene_symbol": [f"Gene_{g:05d}" for g in range(n_genes)],
            "ensembl_id": [f"ENSMUSG{g:011d}" for g in range(n_genes)],
            "mt": [False] * n_genes,
        }
    )
    # Mark first 10 as mitochondrial-like
    var.iloc[:10, var.columns.get_loc("mt")] = True
    var.index = var["gene_symbol"].values

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata
