"""Extract/check pretrained single-cell foundation-model embeddings.

This CLI converts the repo's cache format into AnnData, optionally calls
Geneformer/scGPT/UCE/scFoundation embedding extractors, then runs small
diagnostics on any produced or precomputed embedding matrix.

Example:
    python -m scripts.check_foundation_embeddings \\
        --cache-dir /gscratch/GROUP/data/cache/wmb_isocortex_v1 \\
        --split test \\
        --out-dir /gscratch/GROUP/runs/cellfm/foundation_check \\
        --embedding pca64=/gscratch/GROUP/runs/cellfm/v1/pca64/embeddings_test.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from cellfm.eval.foundation_embedding_checks import embedding_diagnostics
from cellfm.eval.foundation_embedding_io import (
    json_safe,
    load_cache_split,
    load_embedding_matrix,
    parse_named_path,
    payload_to_anndata,
    save_embedding_npz,
)
from cellfm.eval.foundation_model_adapters import run_requested_model

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check pretrained FM embeddings on a cellfm cache split.")
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-cells", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--models",
        nargs="*",
        choices=["geneformer", "scgpt", "uce", "scfoundation"],
        default=[],
        help="External models to run if their packages/checkpoints are installed.",
    )
    p.add_argument(
        "--embedding",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=PATH",
        help="Precomputed embedding file to check. Supports .npz, .npy, .csv, .pt, .h5ad.",
    )
    p.add_argument("--h5ad-obsm-key", default="X_scGPT")
    p.add_argument("--stable-labels", nargs="*", default=["Sst", "Pvalb", "Vip"])
    p.add_argument("--dynamic-label-query", default="OPC")
    p.add_argument("--cycle-score-col", default=None)
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--nproc", type=int, default=4)

    p.add_argument("--scgpt-model-dir", type=Path, default=None)
    p.add_argument("--scgpt-max-length", type=int, default=1200)

    p.add_argument("--geneformer-model-dir", type=Path, default=None)
    p.add_argument("--geneformer-version", default="V2", choices=["V1", "V2"])
    p.add_argument("--geneformer-emb-mode", default="cls", choices=["cls", "cell"])
    p.add_argument("--geneformer-emb-layer", type=int, default=-1)

    p.add_argument("--uce-species", default="mouse")

    p.add_argument("--scfoundation-repo", type=Path, default=None)
    p.add_argument("--scfoundation-model-path", type=Path, default=None)
    p.add_argument("--scfoundation-version", default="ce")
    p.add_argument("--scfoundation-ckpt-name", default="01B-resolution")
    p.add_argument("--scfoundation-pool-type", default="all", choices=["all", "max"])
    p.add_argument("--scfoundation-tgthighres", default="a5")
    p.add_argument("--scfoundation-pre-normalized", default="F", choices=["F", "T", "A"])
    return p


def _check_and_record_embedding(name: str, X_emb, payload, args, *, source: str) -> dict:
    save_embedding_npz(args.out_dir, name, X_emb, payload)
    summary = embedding_diagnostics(
        name,
        X_emb,
        payload,
        stable_labels=args.stable_labels,
        dynamic_label_query=args.dynamic_label_query,
        cycle_score_col=args.cycle_score_col,
        knn_k=args.knn_k,
    )
    summary["source"] = source
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if not args.models and not args.embedding:
        raise SystemExit("Nothing to run. Provide --models and/or --embedding NAME=PATH.")

    payload = load_cache_split(args.cache_dir, args.split, max_cells=args.max_cells, seed=args.seed)
    adata = payload_to_anndata(payload)
    input_h5ad = args.out_dir / "input_sample.h5ad"
    adata.write_h5ad(input_h5ad)
    LOGGER.info("Wrote model input sample: %s", input_h5ad)

    summaries: list[dict] = []
    failures: list[dict[str, str]] = []

    for emb_name, emb_path in args.embedding:
        try:
            X_emb = load_embedding_matrix(emb_path, h5ad_obsm_key=args.h5ad_obsm_key)
            summaries.append(
                _check_and_record_embedding(emb_name, X_emb, payload, args, source=str(emb_path))
            )
        except Exception as exc:
            LOGGER.exception("Embedding check failed for %s", emb_name)
            failures.append({"model": emb_name, "error": str(exc)})

    for model_name in args.models:
        model_work_dir = work_dir / model_name
        model_work_dir.mkdir(parents=True, exist_ok=True)
        try:
            X_emb = run_requested_model(model_name, adata, model_work_dir, args)
            source = str(args.out_dir / model_name / "embeddings.npz")
            summaries.append(
                _check_and_record_embedding(model_name, X_emb, payload, args, source=source)
            )
        except Exception as exc:
            LOGGER.exception("External model extraction failed for %s", model_name)
            failures.append({"model": model_name, "error": str(exc)})

    output = {
        "args": json_safe(vars(args)),
        "input_h5ad": str(input_h5ad),
        "sample": {
            "cache_dir": str(payload.cache_dir),
            "split": payload.split,
            "n_cells": int(payload.y.size),
            "max_cells": args.max_cells,
            "seed": args.seed,
        },
        "summaries": summaries,
        "failures": failures,
    }
    (args.out_dir / "foundation_embedding_check.json").write_text(
        json.dumps(json_safe(output), indent=2)
    )

    if summaries:
        pd.DataFrame(summaries).to_csv(args.out_dir / "foundation_embedding_check.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.out_dir / "foundation_embedding_failures.csv", index=False)

    LOGGER.info("Wrote summary JSON to %s", args.out_dir / "foundation_embedding_check.json")
    if failures:
        LOGGER.warning("%d model(s) failed; see foundation_embedding_failures.csv", len(failures))
    return 0 if summaries else 1


if __name__ == "__main__":
    sys.exit(main())
