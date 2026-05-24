from __future__ import annotations

import numpy as np

from cellfm.data.hvg import select_hvg


def test_select_hvg_returns_unique_indices(synthetic_adata):
    idx = select_hvg(synthetic_adata, n_top=50)
    assert idx.dtype.kind in ("i", "u")
    assert len(idx) <= 50
    assert len(set(idx.tolist())) == len(idx)
    assert idx.min() >= 0
    assert idx.max() < synthetic_adata.n_vars
