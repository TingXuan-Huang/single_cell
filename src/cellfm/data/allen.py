"""Allen-specific data wiring helpers.

The Allen WMB-10X expression-matrix h5ads carry a minimal ``.obs`` indexed by
``cell_label`` but **without** the donor/cluster/subclass annotations that the
rest of this pipeline expects. Those live in the matching
``cell_metadata_with_cluster_annotation.csv`` under
``data/raw/abc/metadata/WMB-10X/<release>/``.

Use :func:`attach_cell_metadata_csv` to left-join that CSV onto an in-memory
AnnData before passing it to ``cellfm.data.qc.run_qc`` /
``cellfm.data.splits.donor_stratified_split``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import pandas as pd

logger = logging.getLogger(__name__)


# Minimal set of columns we need downstream. Anything else from the CSV is fine
# to carry along but these are the load-bearing ones.
DEFAULT_METADATA_COLS: tuple[str, ...] = (
    "donor_label",
    "subclass",
    "class",
    "cluster",
    "region_of_interest_label",
    "region_of_interest_acronym",
    "library_method",
)


def _normalize_cell_label(values) -> pd.Index:
    """Normalize Allen cell labels across h5ad index variants."""
    idx = pd.Index(values).astype(str)
    # anndata.concat(index_unique="-") appends "-<shard key>" to duplicated
    # obs names. Allen cell labels themselves do not contain slashes, while the
    # shard key is a full path in this pipeline.
    idx = idx.str.replace(r"-/.*$", "", regex=True)
    return idx.str.replace(r"\.0$", "", regex=True)


def _candidate_obs_keys(adata: ad.AnnData, preferred: str) -> dict[str, pd.Index]:
    candidates: dict[str, pd.Index] = {}
    if preferred in adata.obs.columns:
        candidates[f"obs.{preferred}"] = _normalize_cell_label(adata.obs[preferred])
    for col in ("cell_label", "cell_id", "cell_barcode", "barcode"):
        if col in adata.obs.columns and f"obs.{col}" not in candidates:
            candidates[f"obs.{col}"] = _normalize_cell_label(adata.obs[col])
    candidates["obs.index"] = _normalize_cell_label(adata.obs.index)
    return candidates


def attach_cell_metadata_csv(
    adata: ad.AnnData,
    csv_path: str | Path,
    *,
    on: str = "cell_label",
    cols: tuple[str, ...] | None = None,
    drop_unmatched: bool = True,
) -> ad.AnnData:
    """Left-join an Allen ``cell_metadata`` CSV onto ``adata.obs``.

    Args:
        adata: AnnData whose ``.obs`` is indexed by ``cell_label``.
        csv_path: Path to ``cell_metadata_with_cluster_annotation.csv``.
        on: Column on which to join. Defaults to ``cell_label``. If the column
            isn't in ``adata.obs`` we fall back to the obs index.
        cols: Subset of CSV columns to attach. Defaults to
            :data:`DEFAULT_METADATA_COLS`. Pass ``("*",)`` to attach everything.
        drop_unmatched: If True (default), drop cells whose metadata row is
            missing. The pipeline assumes every cell has a label, so silent
            NaNs are usually a bug.

    Returns:
        A new AnnData (view-friendly) with the joined columns in ``.obs``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cell metadata CSV not found: {csv_path}")
    meta = pd.read_csv(csv_path)
    logger.info("Loaded cell metadata: %d rows, %d cols", len(meta), meta.shape[1])

    if on not in meta.columns:
        # Try to find a sensible fallback in the CSV.
        for cand in ("cell_label", "cell_id", "cell_barcode", "barcode"):
            if cand in meta.columns:
                meta = meta.rename(columns={cand: on})
                break
        else:
            raise KeyError(
                f"Cannot find join column '{on}' in metadata CSV "
                f"(columns: {list(meta.columns)[:8]}...)"
            )

    keep_cols = list(DEFAULT_METADATA_COLS if cols is None else cols)
    if keep_cols == ["*"]:
        keep_cols = [c for c in meta.columns if c != on]
    missing = [c for c in keep_cols if c not in meta.columns]
    if missing:
        logger.warning(
            "Metadata CSV missing requested columns: %s "
            "(available: %s ...)",
            missing, list(meta.columns)[:10],
        )
        keep_cols = [c for c in keep_cols if c in meta.columns]
    sub = meta[[on, *keep_cols]].copy()
    sub[on] = _normalize_cell_label(sub[on])
    sub = sub.drop_duplicates(subset=[on])
    sub = sub.set_index(on)

    candidates = _candidate_obs_keys(adata, on)
    best_name = ""
    best_keys = None
    best_matches = -1
    for name, keys in candidates.items():
        matches = int(keys.isin(sub.index).sum())
        logger.info("Metadata join candidate %s matched %d / %d cells", name, matches, adata.n_obs)
        if matches > best_matches:
            best_name = name
            best_keys = keys
            best_matches = matches
    if best_keys is None or best_matches == 0:
        raise RuntimeError(
            "Allen metadata join matched 0 cells. "
            f"Tried {list(candidates)} against metadata column '{on}'. "
            f"Example obs index: {list(adata.obs.index[:3].astype(str))}; "
            f"example metadata labels: {list(sub.index[:3].astype(str))}."
        )
    logger.info("Using metadata join key %s (%d / %d matched)", best_name, best_matches, adata.n_obs)

    joined = sub.reindex(best_keys)
    if drop_unmatched:
        matched_mask = joined.notna().all(axis=1).to_numpy()
        n_drop = int((~matched_mask).sum())
        if n_drop > 0:
            logger.warning(
                "Dropping %d / %d cells with missing metadata (likely "
                "filtered out of the Allen cluster annotation).",
                n_drop, adata.n_obs,
            )
            adata = adata[matched_mask].copy()
            joined = joined.loc[matched_mask]

    for c in keep_cols:
        adata.obs[c] = joined[c].to_numpy()
    logger.info(
        "Attached %d metadata columns; resulting AnnData: %d cells.",
        len(keep_cols), adata.n_obs,
    )
    return adata


def filter_to_region(
    adata: ad.AnnData,
    region_keyword: str = "Isocortex",
    region_col: str = "region_of_interest_label",
) -> ad.AnnData:
    """Defensive region filter: keep cells whose ``region_col`` contains
    ``region_keyword``. The Allen isocortex shards already prefilter by region
    but spot-checking after a multi-shard merge is cheap insurance.
    """
    if region_col not in adata.obs.columns:
        logger.warning(
            "No %s column on adata.obs; skipping region filter.", region_col,
        )
        return adata
    mask = adata.obs[region_col].astype(str).str.contains(region_keyword, na=False)
    n_drop = int((~mask).sum())
    if n_drop > 0:
        logger.info(
            "Region filter (%s contains %r): dropping %d / %d cells.",
            region_col, region_keyword, n_drop, adata.n_obs,
        )
        adata = adata[mask.to_numpy()].copy()
    return adata
