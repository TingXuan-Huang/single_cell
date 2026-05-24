"""PCA-64 baseline on log1p-normalized HVG-2k counts.

Provides a reference embedding for kNN-Jaccard and for the per-encoder
comparison table.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def fit_pca64(
    X_train: sparse.csr_matrix, hvg_indices: np.ndarray, n_components: int = 64, seed: int = 0
) -> tuple[TruncatedSVD, np.ndarray]:
    """Fit TruncatedSVD on log1p-normalized HVG counts of the train split."""
    Xh = X_train[:, hvg_indices].astype(np.float32)
    # Per-cell normalize to 1e4 + log1p
    total = np.asarray(Xh.sum(axis=1)).flatten()
    total = np.maximum(total, 1.0)
    Xh = Xh.multiply(1e4 / total[:, None]).log1p()
    svd = TruncatedSVD(n_components=min(n_components, Xh.shape[1] - 1), random_state=seed)
    Z = svd.fit_transform(Xh)
    return svd, Z


def transform_pca64(
    svd: TruncatedSVD, X: sparse.csr_matrix, hvg_indices: np.ndarray
) -> np.ndarray:
    Xh = X[:, hvg_indices].astype(np.float32)
    total = np.asarray(Xh.sum(axis=1)).flatten()
    total = np.maximum(total, 1.0)
    Xh = Xh.multiply(1e4 / total[:, None]).log1p()
    return svd.transform(Xh)
