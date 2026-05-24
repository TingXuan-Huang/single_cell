"""Donor-stratified train/val/test splits.

Locked policy (notes/pipeline_v1_plan.md, Decision #2):
- Split fractions: 70 / 15 / 15 (train / val / test)
- Donor-stratified: no donor appears in two splits
- Class-aware: try to keep subclass distributions balanced across splits
- Seed: 0 (overridable for ablations)
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    donor_col: str = "donor_id"
    label_col: str = "subclass"
    seed: int = 0
    min_donors_per_split: int = 1
    """Hard floor on donors per split. If violated we raise."""


def _check_fracs(cfg: SplitConfig) -> None:
    s = cfg.train_frac + cfg.val_frac + cfg.test_frac
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {s}")


def donor_stratified_split(
    adata: ad.AnnData, cfg: SplitConfig | None = None
) -> dict[str, np.ndarray]:
    """Split cells by donor.

    Returns
    -------
    dict mapping {"train", "val", "test"} -> cell index array (positions in adata).
    Also adds a 'split' column to adata.obs in-place.
    """
    cfg = cfg or SplitConfig()
    _check_fracs(cfg)

    if cfg.donor_col not in adata.obs.columns:
        raise KeyError(
            f"Donor column '{cfg.donor_col}' not in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    rng = np.random.default_rng(cfg.seed)
    donors = pd.Series(adata.obs[cfg.donor_col].astype(str).values)
    donor_to_size = donors.value_counts()
    unique_donors = donor_to_size.index.to_numpy()

    # Greedy allocation: shuffle donors, assign to the split currently most underfilled.
    rng.shuffle(unique_donors)
    target = {
        "train": cfg.train_frac * adata.n_obs,
        "val": cfg.val_frac * adata.n_obs,
        "test": cfg.test_frac * adata.n_obs,
    }
    current = {"train": 0, "val": 0, "test": 0}
    donor_to_split: dict[str, str] = {}

    for donor in unique_donors:
        size = int(donor_to_size[donor])
        # Pick split with largest remaining deficit
        deficits = {k: target[k] - current[k] for k in target}
        chosen = max(deficits, key=lambda k: deficits[k])
        donor_to_split[donor] = chosen
        current[chosen] += size

    split_arr = donors.map(donor_to_split).values
    adata.obs["split"] = pd.Categorical(split_arr, categories=["train", "val", "test"])

    indices = {
        "train": np.where(split_arr == "train")[0],
        "val": np.where(split_arr == "val")[0],
        "test": np.where(split_arr == "test")[0],
    }

    # Floor check on donors per split
    for k, idx in indices.items():
        donors_in_split = adata.obs.iloc[idx][cfg.donor_col].nunique()
        if donors_in_split < cfg.min_donors_per_split:
            raise RuntimeError(
                f"Split '{k}' has only {donors_in_split} donor(s) "
                f"(min required: {cfg.min_donors_per_split}). "
                f"Reduce donor heterogeneity or relax min_donors_per_split."
            )
    return indices


def split_summary(adata: ad.AnnData, cfg: SplitConfig | None = None) -> pd.DataFrame:
    """Per-split cell count, donor count, and subclass coverage."""
    cfg = cfg or SplitConfig()
    if "split" not in adata.obs.columns:
        raise RuntimeError("Run donor_stratified_split first.")
    rows = []
    for sp in ["train", "val", "test"]:
        sub = adata.obs[adata.obs["split"] == sp]
        rows.append(
            {
                "split": sp,
                "n_cells": len(sub),
                "n_donors": sub[cfg.donor_col].nunique(),
                "n_subclasses": sub[cfg.label_col].nunique()
                if cfg.label_col in sub.columns
                else None,
            }
        )
    return pd.DataFrame(rows)
