"""Evaluate one trained run.

Extracts cell embeddings from train (subsample), val, test splits, then computes:
- Linear probe (train -> test) accuracy + macro F1
- kNN probe (k=15, cosine) accuracy + macro F1
- Participation ratio of test embeddings
- NC1 / NC2 / ETF score on test
- Optional kNN-Jaccard vs PCA-64 reference (if --pca-ref is given)

Writes summary.json into the run dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from scripts._config import load_yaml
from scripts.train_one import _build_model, _build_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_checkpoint(run_dir: Path, prefer: str = "best.pt"):
    for name in (prefer, "final.pt"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint found in {run_dir} (looked for best.pt / final.pt).")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate one trained cellfm run.")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Directory created by train_one.py.")
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--encoder-config", required=True, type=Path)
    p.add_argument("--model-config", required=True, type=Path)
    p.add_argument("--checkpoint", default="best.pt", type=str)
    p.add_argument("--max-train-batches", type=int, default=200,
                   help="Limit training-set forward passes for probe-train embeddings.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pca-ref", type=Path, default=None,
                   help="Path to a baseline PCA embedding NPZ produced by "
                        "scripts.build_pca_baseline (uses test split for kNN-Jaccard).")
    args = p.parse_args(argv)

    encoder_cfg = load_yaml(args.encoder_config)
    model_cfg = load_yaml(args.model_config)

    from cellfm.data.cache import CacheManifest
    manifest = CacheManifest.from_json(args.cache_dir / "manifest.json")

    tokenizer = _build_tokenizer(encoder_cfg, manifest)
    model, _ = _build_model(encoder_cfg, model_cfg, manifest, tokenizer)

    ckpt_path = _load_checkpoint(args.run_dir, prefer=args.checkpoint)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Loaded %s on %s", ckpt_path, device)

    from cellfm.training import build_dataloaders
    loaders = build_dataloaders(
        cache_dir=args.cache_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        eval_mode=True,    # disables masking + shuffle for all splits
    )

    from cellfm.eval.extract_embeddings import extract_embeddings
    logger.info("Extracting embeddings (train subsample, val, test) ...")
    emb_train = extract_embeddings(model, loaders["train"], device=device,
                                   max_batches=args.max_train_batches)
    emb_val = extract_embeddings(model, loaders["val"], device=device)
    emb_test = extract_embeddings(model, loaders["test"], device=device)

    # Probes (train -> test)
    from cellfm.eval.probes import knn_probe, linear_probe
    logger.info("Running linear + kNN probes ...")
    probe_lin = linear_probe(emb_train["X"], emb_train["y"], emb_test["X"], emb_test["y"])
    probe_knn = knn_probe(emb_train["X"], emb_train["y"], emb_test["X"], emb_test["y"], k=15)

    # Geometry
    from cellfm.eval.geometry import (
        class_center_distances,
        participation_ratio,
        within_class_variance_trace,
    )
    pr_test = participation_ratio(emb_test["X"])
    wcv_test = within_class_variance_trace(emb_test["X"], emb_test["y"])
    cdist_test = class_center_distances(emb_test["X"], emb_test["y"])

    # Collapse metrics
    from cellfm.metrics import (
        class_center_etf_score,
        nc1_within_class_variability,
        nc2_class_simplex_geometry,
    )
    nc1 = nc1_within_class_variability(emb_test["X"], emb_test["y"])
    nc2 = nc2_class_simplex_geometry(emb_test["X"], emb_test["y"])
    etf = class_center_etf_score(emb_test["X"], emb_test["y"])

    # Optional kNN-Jaccard against PCA-64 reference
    jaccard = float("nan")
    if args.pca_ref is not None and args.pca_ref.exists():
        ref = np.load(args.pca_ref)
        ref_X = ref["X_test"]
        from cellfm.eval.geometry import knn_graph_jaccard
        jaccard = knn_graph_jaccard(emb_test["X"], ref_X, k=15)

    summary = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(ckpt_path),
        "encoder": encoder_cfg.get("name"),
        "size": model_cfg.get("name"),
        "n_test": int(emb_test["X"].shape[0]),
        "d_embedding": int(emb_test["X"].shape[1]),
        **probe_lin,
        **probe_knn,
        "participation_ratio": float(pr_test),
        "within_class_variance_trace": float(wcv_test),
        **cdist_test,
        "nc1": float(nc1),
        "nc2": float(nc2),
        "etf_off_diag_mean": float(etf),
        "knn_jaccard_vs_pca64": float(jaccard),
    }

    out_path = args.run_dir / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    # Save raw test embeddings for downstream re-analysis
    np.savez_compressed(
        args.run_dir / "embeddings_test.npz",
        X_test=emb_test["X"].astype(np.float32),
        y_test=emb_test["y"].astype(np.int64),
    )
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
