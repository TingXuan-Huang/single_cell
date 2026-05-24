"""Geometry / collapse metrics on embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def participation_ratio(X: np.ndarray) -> float:
    """Effective dimensionality: (sum lambda_i)^2 / sum lambda_i^2 (eigenvalues of cov)."""
    if X.shape[0] < 2 or X.shape[1] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc / (Xc.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    num = eigvals.sum() ** 2
    den = (eigvals ** 2).sum()
    return float(num / den) if den > 0 else 0.0


def within_class_variance_trace(X: np.ndarray, y: np.ndarray) -> float:
    """Sum over classes of trace of within-class covariance (lower = more collapsed)."""
    total = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        if Xc.shape[0] < 2:
            continue
        Xc = Xc - Xc.mean(axis=0, keepdims=True)
        total += float((Xc ** 2).sum() / (Xc.shape[0] - 1))
    return total


def class_center_distances(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Average pairwise distance between class centers."""
    centers = []
    for c in np.unique(y):
        Xc = X[y == c]
        if Xc.shape[0] == 0:
            continue
        centers.append(Xc.mean(axis=0))
    if len(centers) < 2:
        return {"mean_center_dist": float("nan")}
    centers = np.stack(centers)
    diffs = centers[:, None, :] - centers[None, :, :]
    d = np.sqrt((diffs ** 2).sum(axis=-1))
    iu = np.triu_indices_from(d, k=1)
    return {"mean_center_dist": float(d[iu].mean())}


def knn_graph_jaccard(
    X_a: np.ndarray, X_b: np.ndarray, k: int = 15, sample: int | None = 5000, seed: int = 0
) -> float:
    """Jaccard similarity between the kNN graphs computed on two embeddings of the SAME cells.

    Useful as a similarity metric versus a reference (e.g. PCA-64). High Jaccard
    means the embedding preserves local neighborhood structure of the reference.
    """
    n = X_a.shape[0]
    if X_b.shape[0] != n:
        raise ValueError("X_a and X_b must have the same number of rows.")
    if n < k + 1:
        return float("nan")

    rng = np.random.default_rng(seed)
    idx = (
        rng.choice(n, size=min(sample or n, n), replace=False)
        if sample and sample < n
        else np.arange(n)
    )
    nbrs_a = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1).fit(X_a)
    nbrs_b = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1).fit(X_b)
    _, ka = nbrs_a.kneighbors(X_a[idx])
    _, kb = nbrs_b.kneighbors(X_b[idx])

    # drop self (column 0)
    ka = ka[:, 1:]
    kb = kb[:, 1:]
    jacs = []
    for i in range(len(idx)):
        sa = set(ka[i].tolist())
        sb = set(kb[i].tolist())
        union = len(sa | sb)
        if union == 0:
            continue
        jacs.append(len(sa & sb) / union)
    return float(np.mean(jacs)) if jacs else float("nan")
