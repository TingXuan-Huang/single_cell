"""Smoke test for the scVI baseline on synthetic data.

Mirrors the role of the PCA baseline in the bake-off: fit on the train split
of the synthetic cache, transform val/test, emit eval_summary.json +
embeddings_test.npz. The whole point is "does the script's I/O contract still
hold" — not "does scVI converge on 400 cells in 1 epoch" — so we use the
smallest possible knobs and assert artifacts exist.

Auto-skipped when scvi-tools isn't installed (e.g. the base CI image without
the [allen] extra). See pyproject.toml.
"""

from __future__ import annotations

import pytest


def test_scvi_baseline_synthetic(synthetic_cache, tmp_path):
    pytest.importorskip("scvi")
    from scripts.build_scvi_baseline import main as scvi_main

    out_dir = tmp_path / "scvi_smoke"
    rc = scvi_main([
        "--cache-dir", str(synthetic_cache),
        "--out-dir", str(out_dir),
        "--max-epochs", "1",
        "--n-latent", "8",
    ])
    assert rc == 0
    assert (out_dir / "eval_summary.json").exists()
    assert (out_dir / "embeddings_test.npz").exists()
