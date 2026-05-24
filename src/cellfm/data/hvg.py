"""Highly variable gene selection.

Computed on the TRAIN split only (locked decision in notes/pipeline_v1_plan.md).
Falls back to a simple variance-on-log1p heuristic when scanpy is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


def select_hvg(
    adata_train: ad.AnnData,
    n_top: int = 2000,
    flavor: str = "seurat_v3",
    layer: str | None = None,
) -> np.ndarray:
    """Return the indices of the top-n highly variable genes.

    Args:
        adata_train: TRAIN split only (do not pass full data; this prevents leakage)
        n_top: how many HVGs to keep
        flavor: 'seurat_v3' if scanpy available, otherwise fallback to variance-of-log1p
        layer: optional .layers key with raw counts

    Returns:
        np.ndarray of gene indices into adata_train.var
    """
    try:
        import scanpy as sc

        adata_local = adata_train.copy()
        sc.pp.highly_variable_genes(
            adata_local,
            n_top_genes=n_top,
            flavor=flavor,
            layer=layer,
            subset=False,
        )
        hvg_mask = adata_local.var["highly_variable"].to_numpy()
        return np.where(hvg_mask)[0]
    except (ImportError, Exception):
        # Fallback: variance of log1p of CPM-ish normalized counts.
        return _select_hvg_fallback(adata_train, n_top=n_top, layer=layer)


def _select_hvg_fallback(adata_train: ad.AnnData, n_top: int, layer: str | None) -> np.ndarray:
    X = adata_train.layers[layer] if layer else adata_train.X
    if sparse.issparse(X):
        X = X.tocsr()
        total = np.asarray(X.sum(axis=1)).flatten().astype(np.float64)
        total = np.maximum(total, 1.0)
        # Per-cell CPM-ish (target sum = 1e4) -> log1p -> variance per gene
        scale = 1e4 / total
        X_norm = X.multiply(scale[:, None])
        # Variance of log1p(X_norm) per column
        # mean
        n = X_norm.shape[0]
        log_X = X_norm.log1p()
        mean = np.asarray(log_X.mean(axis=0)).flatten()
        # E[x^2]: compute via element-wise square
        sq = log_X.power(2)
        ex2 = np.asarray(sq.mean(axis=0)).flatten()
        var = ex2 - mean**2
    else:
        X = np.asarray(X, dtype=np.float64)
        total = np.maximum(X.sum(axis=1, keepdims=True), 1.0)
        X_norm = X * (1e4 / total)
        log_X = np.log1p(X_norm)
        var = log_X.var(axis=0)
    n_top = min(n_top, var.shape[0])
    idx = np.argsort(-var)[:n_top]
    return np.sort(idx)


def save_hvg(hvg_idx: np.ndarray, adata: ad.AnnData, out_path: Path) -> None:
    """Persist HVG selection as JSON with gene symbols and ensembl IDs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_top": int(len(hvg_idx)),
        "indices": hvg_idx.astype(int).tolist(),
        "gene_symbol": adata.var.iloc[hvg_idx].get("gene_symbol", adata.var.index[hvg_idx])
        .astype(str)
        .tolist(),
        "ensembl_id": (
            adata.var.iloc[hvg_idx]["ensembl_id"].astype(str).tolist()
            if "ensembl_id" in adata.var.columns
            else None
        ),
    }
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def load_hvg(in_path: Path) -> np.ndarray:
    with Path(in_path).open() as fh:
        payload = json.load(fh)
    return np.asarray(payload["indices"], dtype=np.int64)
