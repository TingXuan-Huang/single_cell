"""Diagnostics for external single-cell embedding matrices."""

from __future__ import annotations

from typing import Any

import numpy as np

from cellfm.eval.biology import opc_cycle_corr, stable_silhouette
from cellfm.eval.foundation_embedding_io import SplitPayload
from cellfm.eval.geometry import (
    class_center_distances,
    participation_ratio,
    within_class_variance_trace,
)
from cellfm.metrics import (
    class_center_etf_score,
    nc1_within_class_variability,
    nc2_class_simplex_geometry,
)


def _label_substring_mask(labels: np.ndarray, queries: list[str]) -> np.ndarray:
    if not queries:
        return np.zeros(labels.shape[0], dtype=bool)
    lowered = np.char.lower(labels.astype(str))
    mask = np.zeros(labels.shape[0], dtype=bool)
    for query in queries:
        q = query.strip().lower()
        if q:
            mask |= np.char.find(lowered, q) >= 0
    return mask


def _knn_label_purity(X: np.ndarray, y: np.ndarray, k: int) -> float:
    from sklearn.neighbors import NearestNeighbors

    if X.shape[0] <= 2:
        return float("nan")
    k_eff = max(1, min(k, X.shape[0] - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", n_jobs=-1).fit(X)
    _, idx = nbrs.kneighbors(X)
    idx = idx[:, 1:]
    return float((y[idx] == y[:, None]).mean())


def _opc_pc1_variance_ratio(X: np.ndarray, opc_mask: np.ndarray) -> float:
    X_opc = X[opc_mask]
    if X_opc.shape[0] < 3 or X_opc.shape[1] < 2:
        return float("nan")
    Xc = X_opc - X_opc.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    denom = float(np.square(s).sum())
    return float((s[0] ** 2) / denom) if denom > 0 else float("nan")


def embedding_diagnostics(
    name: str,
    X: np.ndarray,
    payload: SplitPayload,
    *,
    stable_labels: list[str],
    dynamic_label_query: str,
    cycle_score_col: str | None,
    knn_k: int,
) -> dict[str, Any]:
    if X.ndim != 2:
        raise ValueError(f"{name}: expected 2D embeddings, got shape {X.shape}")
    if X.shape[0] != payload.y.shape[0]:
        raise ValueError(
            f"{name}: embedding rows ({X.shape[0]}) do not match sampled cells "
            f"({payload.y.shape[0]}). Use the same --split/--max-cells order."
        )
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: embedding matrix contains NaN/Inf values")

    y = payload.y
    labels = payload.label_text.astype(str)
    stable_mask = _label_substring_mask(labels, stable_labels)
    dynamic_mask = _label_substring_mask(labels, [dynamic_label_query])

    cycle_score = None
    if cycle_score_col and cycle_score_col in payload.obs.columns:
        cycle_score = payload.obs[cycle_score_col].to_numpy(dtype=np.float32)

    stable_metrics = (
        stable_silhouette(X, y, subclass_subset=np.unique(y[stable_mask]).astype(int).tolist())
        if stable_mask.any()
        else {"silhouette": float("nan"), "n_cells": 0}
    )
    opc_metrics = opc_cycle_corr(X, dynamic_mask, cycle_score)

    return {
        "model": name,
        "split": payload.split,
        "n_cells": int(X.shape[0]),
        "d_embedding": int(X.shape[1]),
        "n_labels": int(np.unique(y).size),
        "participation_ratio": participation_ratio(X),
        "within_class_variance_trace": within_class_variance_trace(X, y),
        "mean_center_dist": class_center_distances(X, y)["mean_center_dist"],
        "nc1": nc1_within_class_variability(X, y),
        "nc2": nc2_class_simplex_geometry(X, y),
        "etf_off_diag_mean": class_center_etf_score(X, y),
        f"knn{knn_k}_label_purity": _knn_label_purity(X, y, knn_k),
        "stable_label_queries": stable_labels,
        "stable_n_cells": int(stable_metrics["n_cells"]),
        "stable_silhouette": stable_metrics["silhouette"],
        "dynamic_label_query": dynamic_label_query,
        "dynamic_n_cells": int(opc_metrics["n_opc"]),
        "dynamic_pc1_cycle_corr": opc_metrics["opc_pc1_cycle_corr"],
        "dynamic_pc1_variance_ratio": _opc_pc1_variance_ratio(X, dynamic_mask),
        "cycle_score_col": cycle_score_col if cycle_score is not None else None,
    }
