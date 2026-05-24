"""Neural-collapse style metrics (NC1, NC2, ETF-ness).

Reference: Papyan, Han & Donoho, PNAS 2020.

NC1: trace(within-class scatter S_W) / trace(between-class scatter S_B). Lower
     means more collapsed.
NC2: how close class centers are to forming a simplex equiangular tight frame.
     1 - cosine-similarity-spread of normalized class-center vectors.
"""

from __future__ import annotations

import numpy as np


def _class_centers(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.unique(y)
    centers = np.stack([X[y == c].mean(axis=0) for c in classes])
    global_mean = X.mean(axis=0)
    counts = np.array([(y == c).sum() for c in classes])
    return centers, global_mean, counts


def nc1_within_class_variability(X: np.ndarray, y: np.ndarray) -> float:
    """trace(S_W) / trace(S_B). Lower = more collapsed."""
    centers, global_mean, counts = _class_centers(X, y)
    # within-class scatter
    sw = 0.0
    for ci, c in enumerate(np.unique(y)):
        Xc = X[y == c] - centers[ci][None, :]
        sw += float((Xc ** 2).sum())
    sw /= max(1, X.shape[0])
    # between-class scatter
    sb = 0.0
    for ci, c in enumerate(np.unique(y)):
        diff = centers[ci] - global_mean
        sb += float(counts[ci] * (diff ** 2).sum())
    sb /= max(1, X.shape[0])
    return float(sw / max(1e-12, sb))


def nc2_class_simplex_geometry(X: np.ndarray, y: np.ndarray) -> float:
    """Spread of pairwise cosine similarities of (center - global mean) vectors.

    Lower = centers more equiangular (closer to simplex ETF target).
    """
    centers, global_mean, _ = _class_centers(X, y)
    diffs = centers - global_mean[None, :]
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    u = diffs / norms
    cos = u @ u.T
    iu = np.triu_indices_from(cos, k=1)
    if len(iu[0]) == 0:
        return float("nan")
    return float(np.std(cos[iu]))


def class_center_etf_score(X: np.ndarray, y: np.ndarray) -> float:
    """Convenience: mean off-diagonal cosine of (center - global mean) directions.

    For perfect ETF with K classes this approaches -1/(K-1).
    """
    centers, global_mean, _ = _class_centers(X, y)
    diffs = centers - global_mean[None, :]
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    u = diffs / norms
    cos = u @ u.T
    iu = np.triu_indices_from(cos, k=1)
    return float(cos[iu].mean()) if len(iu[0]) > 0 else float("nan")
