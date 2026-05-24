"""Biology-flavored metrics.

- Stable-state test: silhouette score on SST/PV/VIP (or any user-named subclasses).
- Dynamic-state test: correlation between PC1 of OPC embeddings and a cell-cycle
  score column in obs (if available).

These run on a SUBSET of cells filtered by subclass label or column value.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def stable_silhouette(
    X: np.ndarray,
    y: np.ndarray,
    *,
    subclass_subset: list[int] | None = None,
) -> dict[str, float]:
    """Silhouette on the given subclasses (default: all)."""
    if subclass_subset is not None:
        mask = np.isin(y, subclass_subset)
        Xs, ys = X[mask], y[mask]
    else:
        Xs, ys = X, y
    if Xs.shape[0] < 4 or len(np.unique(ys)) < 2:
        return {"silhouette": float("nan"), "n_cells": int(Xs.shape[0])}
    score = float(silhouette_score(Xs, ys, metric="cosine", sample_size=min(5000, Xs.shape[0])))
    return {"silhouette": score, "n_cells": int(Xs.shape[0])}


def opc_cycle_corr(
    X: np.ndarray,
    opc_mask: np.ndarray,
    cycle_score: np.ndarray | None,
) -> dict[str, float]:
    """Correlation between PC1 of OPC embeddings and a cell-cycle score.

    If `cycle_score` is None we return NaN (the caller should compute it
    upstream with scanpy.tl.score_genes_cell_cycle or similar).
    """
    if cycle_score is None:
        return {"opc_pc1_cycle_corr": float("nan"), "n_opc": int(opc_mask.sum())}
    Xopc = X[opc_mask]
    cscore = np.asarray(cycle_score)[opc_mask]
    if Xopc.shape[0] < 3 or Xopc.shape[1] < 1:
        return {"opc_pc1_cycle_corr": float("nan"), "n_opc": int(opc_mask.sum())}

    Xc = Xopc - Xopc.mean(axis=0, keepdims=True)
    # PC1 via SVD
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Xc @ vt[0]
    if np.std(pc1) < 1e-9 or np.std(cscore) < 1e-9:
        return {"opc_pc1_cycle_corr": float("nan"), "n_opc": int(opc_mask.sum())}
    r = float(np.corrcoef(pc1, cscore)[0, 1])
    return {"opc_pc1_cycle_corr": r, "n_opc": int(opc_mask.sum())}
