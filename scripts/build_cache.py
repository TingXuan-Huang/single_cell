"""Build the preprocessed cache for an AnnData input.

Pipeline:
  load AnnData -> QC -> donor-stratified split -> HVG (train only) ->
  choose L (95th pctile nnz, rounded up to power of 2) -> write_cache.

Supports two input modes:
  1. --input-h5ad PATH   (real Allen data on Hyak)
  2. --synthetic         (generate from cellfm.data.synthetic for smoke tests)

Both honor a data config YAML for thresholds and the output cache_dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from scripts._config import load_yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_adata(args, cfg):
    if args.synthetic:
        from cellfm.data.synthetic import make_synthetic_anndata

        sy = cfg.get("synthetic", {})
        logger.info("Generating synthetic AnnData with %s", sy)
        return make_synthetic_anndata(**sy)

    import glob

    import anndata as ad

    # Resolve input(s). Three options, checked in order:
    #   --input-h5ad PATH (single file)
    #   --input-h5ad-glob PATTERN (one-or-many shards, concatenated)
    #   cfg["input_h5ad"] or cfg["input_h5ad_glob"]
    glob_pat = args.input_h5ad_glob or cfg.get("input_h5ad_glob")
    if glob_pat:
        paths = sorted(glob.glob(str(glob_pat)))
        if not paths:
            raise FileNotFoundError(f"No h5ad files matched glob: {glob_pat}")
        if len(paths) == 1:
            logger.info("Reading single AnnData from %s ...", paths[0])
            return ad.read_h5ad(paths[0])
        logger.info("Reading + concatenating %d AnnData shards:", len(paths))
        for p in paths:
            logger.info("  %s", p)
        shards = [ad.read_h5ad(p) for p in paths]
        # join='outer' is robust when shards have slightly different .var rows;
        # for WMB-10X the gene tables match so this is a no-op cost-wise.
        merged = ad.concat(shards, axis=0, join="outer", merge="same",
                           label="source_shard", keys=paths,
                           index_unique="-")
        return merged

    path = args.input_h5ad or cfg.get("input_h5ad")
    if path is None:
        raise FileNotFoundError(
            "No AnnData input specified. Provide --input-h5ad, "
            "--input-h5ad-glob, or set input_h5ad / input_h5ad_glob in the YAML."
        )
    logger.info("Reading AnnData from %s ...", path)
    return ad.read_h5ad(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build preprocessed cellfm cache.")
    p.add_argument("--data-config", required=True, type=Path)
    p.add_argument("--input-h5ad", type=Path, default=None,
                   help="Single AnnData h5ad file. Overrides input_h5ad in YAML.")
    p.add_argument("--input-h5ad-glob", type=str, default=None,
                   help="Glob pattern for AnnData shards; all matches will be "
                   "concatenated along the cell axis. Use this for the "
                   "per-region Allen WMB-10X downloads.")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--metadata-csv", type=Path, default=None,
                   help="Path to Allen cell_metadata_with_cluster_annotation.csv. "
                   "If provided, columns like donor_label / subclass are joined "
                   "into adata.obs before QC + splitting.")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic AnnData.")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_yaml(args.data_config)
    if args.cache_dir is not None:
        cfg["cache_dir"] = str(args.cache_dir)
    if args.seed is not None:
        cfg.setdefault("splits", {})["seed"] = int(args.seed)

    adata = _load_adata(args, cfg)
    logger.info("AnnData: %d cells x %d genes", adata.n_obs, adata.n_vars)

    # 0) (Allen only) Attach cell_metadata_with_cluster_annotation if the
    # config provides it. The Allen expression-matrix h5ads ship without
    # donor/subclass annotations -- those need to be left-joined from the
    # metadata CSV before QC and splitting.
    metadata_csv = args.metadata_csv or cfg.get("metadata_csv")
    if metadata_csv:
        from cellfm.data.allen import attach_cell_metadata_csv, filter_to_region

        adata = attach_cell_metadata_csv(adata, metadata_csv)
        if cfg.get("region_filter"):
            adata = filter_to_region(
                adata,
                region_keyword=cfg["region_filter"].get("keyword", "Isocortex"),
                region_col=cfg["region_filter"].get("column", "region_of_interest_label"),
            )
        logger.info(
            "After metadata attach: %d cells x %d genes",
            adata.n_obs, adata.n_vars,
        )

    # 1) QC
    from cellfm.data.qc import (
        QCConfig,
        compute_nnz_distribution,
        pick_L_power_of_two,
        run_qc,
    )
    qc_cfg = cfg.get("qc", {})
    qc = QCConfig(
        min_nnz=int(qc_cfg.get("min_nnz", 500)),
        max_nnz=qc_cfg.get("max_nnz"),
        min_total_counts=int(qc_cfg.get("min_total_counts", 1000)),
        max_pct_mt=float(qc_cfg.get("max_pct_mt", 10.0)),
        max_doublet_score=qc_cfg.get("max_doublet_score"),
    )
    adata, qc_report = run_qc(adata, qc)
    logger.info("\n%s", qc_report.summary())

    # 2) Splits (donor-stratified)
    from cellfm.data.splits import SplitConfig, donor_stratified_split, split_summary

    sp = cfg.get("splits", {})
    # train_frac = 1 - val - test
    val_frac = float(sp.get("val_frac", 0.15))
    test_frac = float(sp.get("test_frac", 0.15))
    train_frac = max(0.0, 1.0 - val_frac - test_frac)
    sp_cfg = SplitConfig(
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        donor_col=str(sp.get("donor_col", "donor_id")),
        label_col=str(cfg.get("label_col", "subclass")),
        seed=int(sp.get("seed", 0)),
    )
    splits = donor_stratified_split(adata, sp_cfg)
    logger.info("\nSplit summary:\n%s", split_summary(adata, sp_cfg))

    # 3) Choose L from train-split nnz
    seq_cfg = cfg.get("seq_length", {})
    adata_train = adata[splits["train"]]
    nnz_stats = compute_nnz_distribution(adata_train)
    logger.info("Train-split nnz stats: %s", nnz_stats)
    pctile = int(seq_cfg.get("pctile", 95))
    target = nnz_stats[f"p{pctile:02d}"]
    L = pick_L_power_of_two(target) if seq_cfg.get("power_of_two", True) else target
    L = max(int(seq_cfg.get("cap_min", 128)), min(int(seq_cfg.get("cap_max", 2048)), L))
    logger.info("Chose L=%d (pctile=%d -> %d, capped to [%d, %d])",
                L, pctile, target,
                seq_cfg.get("cap_min", 128), seq_cfg.get("cap_max", 2048))

    # 4) HVG (train only)
    from cellfm.data.hvg import select_hvg
    hvg_cfg = cfg.get("hvg", {})
    # Map config method -> scanpy flavor. If scanpy is unavailable or the flavor
    # call fails at runtime, select_hvg() transparently falls back to the
    # variance-of-log1p heuristic.
    method = str(hvg_cfg.get("method", "scanpy_seurat_v3"))
    flavor = {
        "scanpy_seurat_v3": "seurat_v3",
        "scanpy_seurat": "seurat",
        "scanpy_cell_ranger": "cell_ranger",
        "variance": "seurat_v3",
    }.get(method, "seurat_v3")
    hvg_idx = select_hvg(adata_train, n_top=int(hvg_cfg.get("n_top", 2000)), flavor=flavor)
    logger.info("Selected %d HVGs (method=%s -> flavor=%s).", len(hvg_idx), method, flavor)

    # 5) Write cache
    from cellfm.data.cache import write_cache

    cache_dir = Path(cfg["cache_dir"])
    qc_summary = {
        "n_in": qc_report.n_in,
        "n_out": qc_report.n_out,
        "dropped_min_nnz": qc_report.dropped_min_nnz,
        "dropped_max_nnz": qc_report.dropped_max_nnz,
        "dropped_min_total_counts": qc_report.dropped_min_total_counts,
        "dropped_max_pct_mt": qc_report.dropped_max_pct_mt,
        "dropped_max_doublet": qc_report.dropped_max_doublet,
        "median_nnz_after": qc_report.details.get("median_nnz_after"),
        "median_total_after": qc_report.details.get("median_total_after"),
        "nnz_stats_train": nnz_stats,
    }

    manifest = write_cache(
        adata,
        split_indices=splits,
        hvg_indices=hvg_idx,
        L=L,
        label_col=str(cfg.get("label_col", "subclass")),
        out_dir=cache_dir,
        qc_summary=qc_summary,
    )

    # Write a small audit doc
    (cache_dir / "BUILD_AUDIT.json").write_text(json.dumps(
        {
            "data_config": str(args.data_config),
            "synthetic": bool(args.synthetic),
            "L": int(L),
            "hvg_n": int(len(hvg_idx)),
            "splits": manifest.splits,
            "qc": qc_summary,
        }, indent=2,
    ))
    logger.info("Wrote cache + manifest to %s", cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
