"""Aggregate eval_summary.json files from all runs into a single markdown table.

Usage:
    python -m scripts.build_comparison_table \\
        --run-roots /gscratch/.../runs/cellfm/v1 \\
        --pca-baseline /gscratch/.../runs/cellfm/v1/pca64 \\
        --scvi-baseline /gscratch/.../runs/cellfm/v1/scvi32 \\
        --out  /gscratch/.../runs/cellfm/v1/COMPARISON.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Transformer-body parameter counts from notes/pipeline_v1_plan.md §M1.
# Baselines (pca64, scvi) are not in this map: their rows get NaN slope.
SIZE_TO_BODY_PARAMS: dict[str, float] = {
    "tiny_1m": 0.9e6,
    "tiny_3m": 2.2e6,
    "tiny_5m": 5.2e6,
    "tiny_10m": 10.5e6,
}


def _collect(run_root: Path) -> list[dict]:
    rows = []
    for s in sorted(run_root.glob("*/eval_summary.json")):
        try:
            row = json.loads(s.read_text())
            row["run_id"] = s.parent.name
            rows.append(row)
        except Exception as e:
            logger.warning("Skipping %s (%s)", s, e)
    return rows


def _add_scaling_slope(df: pd.DataFrame) -> pd.DataFrame:
    """Per encoder, fit linear_acc ~ a + b*log2(body_params); broadcast b to each row.

    This is the load-bearing column for justifying the 4-point size sweep: a
    positive slope means the encoder benefits from scale, and the magnitude
    lets us rank encoders by scaling efficiency rather than absolute accuracy.
    Encoders with <2 sized runs (e.g. baselines) get NaN.
    """
    df = df.copy()
    df["slope_acc_per_log2_params"] = float("nan")
    if not {"encoder", "size", "linear_acc"}.issubset(df.columns):
        return df
    for encoder, grp in df.groupby("encoder", dropna=True):
        body = grp["size"].map(SIZE_TO_BODY_PARAMS)
        mask = body.notna() & grp["linear_acc"].notna()
        if mask.sum() < 2:
            continue
        x = np.log2(body[mask].to_numpy(dtype=float))
        y = grp.loc[mask, "linear_acc"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        df.loc[df["encoder"] == encoder, "slope_acc_per_log2_params"] = slope
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-roots", nargs="+", required=True, type=Path)
    p.add_argument("--pca-baseline", type=Path, default=None,
                   help="Optional dir containing one PCA-baseline eval_summary.json.")
    p.add_argument("--scvi-baseline", type=Path, default=None,
                   help="Optional dir containing one scVI-baseline eval_summary.json.")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    rows: list[dict] = []
    for root in args.run_roots:
        rows.extend(_collect(root))
    for baseline_dir, run_id in (
        (args.pca_baseline, "pca64"),
        (args.scvi_baseline, "scvi"),
    ):
        if baseline_dir is None:
            continue
        sj = baseline_dir / "eval_summary.json"
        if sj.exists():
            row = json.loads(sj.read_text())
            row["run_id"] = run_id
            rows.append(row)

    if not rows:
        logger.error("No eval_summary.json files found.")
        return 1

    df = pd.DataFrame(rows)
    df = _add_scaling_slope(df)
    # Reorder columns
    front = [
        "run_id", "encoder", "size",
        "linear_acc", "linear_macro_f1", "slope_acc_per_log2_params",
        "knn15_acc", "knn15_macro_f1",
        "participation_ratio", "nc1", "nc2", "etf_off_diag_mean",
        "knn_jaccard_vs_pca64",
        "within_class_variance_trace", "mean_center_dist",
        "n_test", "d_embedding",
    ]
    cols = [c for c in front if c in df.columns] + [
        c for c in df.columns if c not in front
    ]
    df = df[cols].sort_values(["encoder", "size"], kind="stable", na_position="last")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("# Encoder Comparison\n\n" + df.to_markdown(index=False) + "\n")
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    logger.info("Wrote %s and %s", args.out, args.out.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
