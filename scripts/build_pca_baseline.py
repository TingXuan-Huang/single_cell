"""Fit PCA-64 on log1p-normalized HVG-2k counts of the train split,
transform val/test, run the same probes + geometry metrics, save outputs.

Outputs: <out_dir>/{summary.json, embeddings_test.npz}.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_split_X(cache_dir: Path, split: str) -> sparse.csr_matrix:
    return sparse.load_npz(cache_dir / f"{split}_X.npz").tocsr()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PCA-64 baseline.")
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-components", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from cellfm.data.cache import CacheManifest
    manifest = CacheManifest.from_json(args.cache_dir / "manifest.json")
    hvg = np.asarray(manifest.hvg_indices, dtype=np.int64)
    n_classes = len(manifest.label_vocab)
    logger.info("Manifest: n_genes=%d n_hvg=%d n_classes=%d", manifest.n_genes, hvg.size, n_classes)

    import pandas as pd

    def load_obs_y(split: str) -> np.ndarray:
        obs = pd.read_parquet(args.cache_dir / f"{split}_obs.parquet")
        return obs["label"].to_numpy().astype(np.int64)

    X_train = _load_split_X(args.cache_dir, "train")
    X_val = _load_split_X(args.cache_dir, "val")
    X_test = _load_split_X(args.cache_dir, "test")
    y_train = load_obs_y("train")
    y_val = load_obs_y("val")
    y_test = load_obs_y("test")

    from cellfm.eval.pca_baseline import fit_pca64, transform_pca64
    logger.info("Fitting PCA-%d on log1p HVG train ...", args.n_components)
    svd, Z_train = fit_pca64(X_train, hvg, n_components=args.n_components, seed=args.seed)
    Z_val = transform_pca64(svd, X_val, hvg)
    Z_test = transform_pca64(svd, X_test, hvg)

    # Probes
    from cellfm.eval.probes import knn_probe, linear_probe
    logger.info("Running linear + kNN probes ...")
    probe_lin = linear_probe(Z_train, y_train, Z_test, y_test, seed=args.seed)
    probe_knn = knn_probe(Z_train, y_train, Z_test, y_test, k=15)

    # Geometry / collapse
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

    summary = {
        "encoder": "pca64",
        "size": "n/a",
        "n_components": int(args.n_components),
        "explained_var_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "n_test": int(Z_test.shape[0]),
        "d_embedding": int(Z_test.shape[1]),
        **probe_lin,
        **probe_knn,
        "participation_ratio": float(participation_ratio(Z_test)),
        "within_class_variance_trace": float(within_class_variance_trace(Z_test, y_test)),
        **class_center_distances(Z_test, y_test),
        "nc1": float(nc1_within_class_variability(Z_test, y_test)),
        "nc2": float(nc2_class_simplex_geometry(Z_test, y_test)),
        "etf_off_diag_mean": float(class_center_etf_score(Z_test, y_test)),
    }

    (args.out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.out_dir / "embeddings_test.npz",
        X_test=Z_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
        X_val=Z_val.astype(np.float32),
        y_val=y_val.astype(np.int64),
    )
    logger.info("Wrote %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
