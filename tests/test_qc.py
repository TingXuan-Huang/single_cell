from __future__ import annotations

import numpy as np

from cellfm.data.qc import (
    QCConfig,
    compute_nnz_distribution,
    pick_L_power_of_two,
    run_qc,
)


def test_run_qc_drops_low_nnz(synthetic_adata):
    cfg = QCConfig(min_nnz=200, min_total_counts=0, max_pct_mt=100.0, max_doublet_score=None)
    a2, r = run_qc(synthetic_adata, cfg)
    assert r.n_out <= r.n_in
    assert a2.n_obs == r.n_out


def test_run_qc_keeps_all_when_loose(synthetic_adata):
    cfg = QCConfig(min_nnz=0, min_total_counts=0, max_pct_mt=200.0, max_doublet_score=None)
    a2, r = run_qc(synthetic_adata, cfg)
    assert r.n_out == synthetic_adata.n_obs


def test_nnz_distribution_keys(synthetic_adata):
    d = compute_nnz_distribution(synthetic_adata)
    for k in ("min", "p05", "p50", "p95", "p99", "max", "mean"):
        assert k in d


def test_pick_L_power_of_two():
    assert pick_L_power_of_two(1) == 1
    assert pick_L_power_of_two(2) == 2
    assert pick_L_power_of_two(3) == 4
    assert pick_L_power_of_two(1500) == 2048
    assert pick_L_power_of_two(1024) == 1024
