"""Cell-level QC filters for WMB-10X-like AnnData.

Designed to be a thin, transparent layer. Each filter records what it dropped
and why so reports are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
from scipy import sparse


@dataclass
class QCConfig:
    """QC thresholds. Defaults are conservative for WMB-10X-like data."""

    min_nnz: int = 500
    """Minimum number of detected genes per cell."""

    max_nnz: int | None = None
    """Maximum number of detected genes per cell. None = no cap."""

    min_total_counts: int = 1000
    """Minimum total UMI per cell."""

    max_pct_mt: float = 10.0
    """Maximum mitochondrial gene percentage."""

    max_doublet_score: float | None = 0.3
    """Maximum doublet score, if available. None disables this filter."""

    mt_var_col: str = "mt"
    """Column in adata.var marking mitochondrial genes."""

    pct_mt_obs_col: str | None = "pct_mt"
    """Column in adata.obs holding precomputed mt%; if None we compute it."""

    doublet_obs_col: str | None = "doublet_score"
    """Column in adata.obs holding doublet scores; if None the filter is skipped."""


@dataclass
class QCReport:
    """Audit log of what QC dropped."""

    n_in: int = 0
    n_out: int = 0
    dropped_min_nnz: int = 0
    dropped_max_nnz: int = 0
    dropped_min_total_counts: int = 0
    dropped_max_pct_mt: int = 0
    dropped_max_doublet: int = 0
    details: dict[str, float | int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"QC: {self.n_in:,} cells in, {self.n_out:,} cells kept",
            f"  dropped (min_nnz):           {self.dropped_min_nnz:,}",
            f"  dropped (max_nnz):           {self.dropped_max_nnz:,}",
            f"  dropped (min_total_counts):  {self.dropped_min_total_counts:,}",
            f"  dropped (max_pct_mt):        {self.dropped_max_pct_mt:,}",
            f"  dropped (max_doublet):       {self.dropped_max_doublet:,}",
        ]
        for k, v in self.details.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def _ensure_csr(X) -> sparse.csr_matrix:
    if sparse.issparse(X):
        return X.tocsr()
    return sparse.csr_matrix(X)


def _per_cell_stats(adata: ad.AnnData, cfg: QCConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nnz, total_counts, pct_mt) per cell."""
    X = _ensure_csr(adata.X)
    nnz = np.asarray((X > 0).sum(axis=1)).flatten()
    total = np.asarray(X.sum(axis=1)).flatten().astype(np.float64)

    if cfg.pct_mt_obs_col and cfg.pct_mt_obs_col in adata.obs.columns:
        pct_mt = adata.obs[cfg.pct_mt_obs_col].to_numpy().astype(np.float32)
    elif cfg.mt_var_col in adata.var.columns:
        mt_mask = adata.var[cfg.mt_var_col].to_numpy().astype(bool)
        mt_counts = np.asarray(X[:, mt_mask].sum(axis=1)).flatten().astype(np.float64)
        safe_total = np.maximum(total, 1.0)
        pct_mt = (mt_counts / safe_total * 100.0).astype(np.float32)
    else:
        pct_mt = np.zeros(adata.n_obs, dtype=np.float32)

    return nnz, total, pct_mt


def run_qc(adata: ad.AnnData, cfg: QCConfig | None = None) -> tuple[ad.AnnData, QCReport]:
    """Apply QC filters and return a filtered AnnData copy + an audit report.

    Returns
    -------
    adata_qc: AnnData with bad cells removed (copy)
    report: QCReport
    """
    cfg = cfg or QCConfig()
    report = QCReport(n_in=adata.n_obs)

    nnz, total, pct_mt = _per_cell_stats(adata, cfg)

    keep = np.ones(adata.n_obs, dtype=bool)

    bad = nnz < cfg.min_nnz
    report.dropped_min_nnz = int(bad.sum())
    keep &= ~bad

    if cfg.max_nnz is not None:
        bad = nnz > cfg.max_nnz
        report.dropped_max_nnz = int((bad & keep).sum())
        keep &= ~bad

    bad = total < cfg.min_total_counts
    report.dropped_min_total_counts = int((bad & keep).sum())
    keep &= ~bad

    bad = pct_mt > cfg.max_pct_mt
    report.dropped_max_pct_mt = int((bad & keep).sum())
    keep &= ~bad

    if cfg.max_doublet_score is not None and cfg.doublet_obs_col in adata.obs.columns:
        dscore = adata.obs[cfg.doublet_obs_col].to_numpy().astype(np.float32)
        bad = dscore > cfg.max_doublet_score
        report.dropped_max_doublet = int((bad & keep).sum())
        keep &= ~bad

    adata_qc = adata[keep].copy()
    report.n_out = int(keep.sum())
    report.details["median_nnz_after"] = float(np.median(nnz[keep])) if keep.any() else 0.0
    report.details["median_total_after"] = float(np.median(total[keep])) if keep.any() else 0.0
    return adata_qc, report


def compute_nnz_distribution(adata: ad.AnnData) -> dict:
    """Return per-cell nnz statistics for picking sequence length L."""
    X = _ensure_csr(adata.X)
    nnz = np.asarray((X > 0).sum(axis=1)).flatten()
    return {
        "min": int(nnz.min()),
        "p05": int(np.percentile(nnz, 5)),
        "p50": int(np.percentile(nnz, 50)),
        "p75": int(np.percentile(nnz, 75)),
        "p90": int(np.percentile(nnz, 90)),
        "p95": int(np.percentile(nnz, 95)),
        "p99": int(np.percentile(nnz, 99)),
        "max": int(nnz.max()),
        "mean": float(nnz.mean()),
    }


def pick_L_power_of_two(p95: int) -> int:
    """Round 95th percentile up to the next power of two for token-sequence L."""
    L = 1
    while L < p95:
        L *= 2
    return L
