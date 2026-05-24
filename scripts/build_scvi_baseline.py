"""Fit scVI on raw HVG-2k counts of the train split, transform val/test,
run the same probes + geometry metrics as the encoder runs, save outputs.

scVI is the field-standard cell VAE (ZINB likelihood, β scheduling, library
size factor). It lives in the ``baselines`` slot of the bake-off — parallel to
PCA-64 — because its architecture ("compress then dense") is distinct from the
Direction-C "tokenize then sequence model" encoders being compared.

Outputs (mirrors PCA baseline):
    <out_dir>/
        eval_summary.json
        embeddings_test.npz  # X_test, y_test, X_val, y_val (float32, int64)
        scvi_model/          # full scvi-tools checkpoint dir
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


def _adata_for_split(
    cache_dir: Path, split: str, hvg_indices: np.ndarray
):
    """Build a minimal AnnData(HVG-only, raw counts) for scVI."""
    import anndata as ad
    import pandas as pd

    X = _load_split_X(cache_dir, split)[:, hvg_indices].astype(np.float32)
    obs = pd.read_parquet(cache_dir / f"{split}_obs.parquet").reset_index(drop=True)
    return ad.AnnData(X=X, obs=obs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="scVI baseline (HVG-2k, raw counts).")
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-latent", type=int, default=32,
                   help="VAE latent dimension. Default 32 matches Track B plan.")
    p.add_argument("--n-hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--batch-key", type=str, default=None,
                   help="obs column to use as scVI batch covariate (e.g. donor_id). "
                        "If omitted, scVI is fit without batch correction.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from cellfm.data.cache import CacheManifest
    manifest = CacheManifest.from_json(args.cache_dir / "manifest.json")
    hvg = np.asarray(manifest.hvg_indices, dtype=np.int64)
    n_classes = len(manifest.label_vocab)
    logger.info("Manifest: n_genes=%d n_hvg=%d n_classes=%d",
                manifest.n_genes, hvg.size, n_classes)

    try:
        import scvi  # noqa: F401
        import torch
    except ImportError as e:
        logger.error("scvi-tools not installed. `pip install -e .[allen]` first. (%s)", e)
        return 2

    # Build AnnData objects from the cache for each split.
    adata_train = _adata_for_split(args.cache_dir, "train", hvg)
    adata_val = _adata_for_split(args.cache_dir, "val", hvg)
    adata_test = _adata_for_split(args.cache_dir, "test", hvg)

    y_train = adata_train.obs["label"].to_numpy().astype(np.int64)
    y_val = adata_val.obs["label"].to_numpy().astype(np.int64)
    y_test = adata_test.obs["label"].to_numpy().astype(np.int64)

    logger.info("Setting up scVI on train (%d cells, %d HVGs)...",
                adata_train.n_obs, adata_train.n_vars)
    scvi.settings.seed = int(args.seed)

    setup_kwargs: dict = {}
    if args.batch_key is not None:
        if args.batch_key not in adata_train.obs.columns:
            logger.warning("--batch-key=%s not in obs; falling back to no batch correction.",
                           args.batch_key)
        else:
            setup_kwargs["batch_key"] = args.batch_key

    scvi.model.SCVI.setup_anndata(adata_train, **setup_kwargs)
    model = scvi.model.SCVI(
        adata_train,
        n_latent=int(args.n_latent),
        n_hidden=int(args.n_hidden),
        n_layers=int(args.n_layers),
    )

    # Train. scvi-tools uses Lightning under the hood; max_epochs is the main knob.
    logger.info("Training scVI for %d epochs ...", args.max_epochs)
    train_kwargs: dict = {"max_epochs": int(args.max_epochs)}
    # Honor GPU availability without requiring it (e.g. for `--max-epochs 1` smoke).
    if torch.cuda.is_available():
        train_kwargs["accelerator"] = "gpu"
        train_kwargs["devices"] = 1
    else:
        train_kwargs["accelerator"] = "cpu"
    model.train(**train_kwargs)

    # For val/test transforms, scvi-tools wants the same setup applied.
    # Easiest portable path: pass adata_val/adata_test through .get_latent_representation
    # after registering them with the trained model. Newer API: `prepare_query_anndata`.
    for ad_ in (adata_val, adata_test):
        try:
            scvi.model.SCVI.prepare_query_anndata(ad_, model)
        except AttributeError:
            # Older scvi-tools: rely on identical var ordering (which we have).
            pass

    logger.info("Encoding latents ...")
    Z_train = model.get_latent_representation(adata_train).astype(np.float32)
    Z_val = model.get_latent_representation(adata_val).astype(np.float32)
    Z_test = model.get_latent_representation(adata_test).astype(np.float32)

    # Same probes/geometry suite as PCA baseline + encoder runs.
    from cellfm.eval.probes import knn_probe, linear_probe
    logger.info("Running linear + kNN probes ...")
    probe_lin = linear_probe(Z_train, y_train, Z_test, y_test, seed=args.seed)
    probe_knn = knn_probe(Z_train, y_train, Z_test, y_test, k=15)

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
        "encoder": "scvi",
        "size": f"zd{int(args.n_latent)}",
        "n_latent": int(args.n_latent),
        "n_hidden": int(args.n_hidden),
        "n_layers": int(args.n_layers),
        "max_epochs": int(args.max_epochs),
        "batch_key": args.batch_key,
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
    model.save(str(args.out_dir / "scvi_model"), overwrite=True)
    logger.info("Wrote %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
