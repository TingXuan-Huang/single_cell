from __future__ import annotations

import json

import numpy as np

from scripts import check_foundation_embeddings as checker


def test_precomputed_embedding_check_smoke(synthetic_cache, tmp_path):
    payload = checker.load_cache_split(
        synthetic_cache,
        "test",
        max_cells=32,
        seed=0,
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(payload.y.shape[0], 12)).astype(np.float32)
    emb_path = tmp_path / "dummy_embeddings.npz"
    np.savez_compressed(emb_path, X_test=X)

    out_dir = tmp_path / "foundation_check"
    rc = checker.main(
        [
            "--cache-dir",
            str(synthetic_cache),
            "--split",
            "test",
            "--out-dir",
            str(out_dir),
            "--max-cells",
            "32",
            "--embedding",
            f"dummy={emb_path}",
        ]
    )

    assert rc == 0
    summary_path = out_dir / "foundation_embedding_check.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["summaries"][0]["model"] == "dummy"
    assert payload["summaries"][0]["n_cells"] == X.shape[0]
    assert payload["summaries"][0]["d_embedding"] == X.shape[1]
    assert (out_dir / "dummy" / "embeddings.npz").exists()


def test_embedding_row_mismatch_is_reported(synthetic_cache, tmp_path):
    emb_path = tmp_path / "bad_embeddings.npy"
    np.save(emb_path, np.zeros((2, 4), dtype=np.float32))

    out_dir = tmp_path / "foundation_check"
    rc = checker.main(
        [
            "--cache-dir",
            str(synthetic_cache),
            "--split",
            "test",
            "--out-dir",
            str(out_dir),
            "--max-cells",
            "32",
            "--embedding",
            f"bad={emb_path}",
        ]
    )

    assert rc == 1
    failures = (out_dir / "foundation_embedding_failures.csv").read_text()
    assert "embedding rows" in failures
