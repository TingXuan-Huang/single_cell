"""Preprocessed-shard cache layer.

Goal: one canonical on-disk format that every encoder reads. Random access by
cell index must be fast (~ms) without loading the whole matrix.

Layout per split:

  data/cache/<split>_X.npz       # scipy sparse CSR (cells x genes), int32 counts
  data/cache/<split>_obs.parquet # per-cell metadata
  data/cache/manifest.json       # gene vocab, HVG indices, L, label vocab, etc.

Loaders:
- CellShardDataset: torch Dataset on top of CSR + obs
- get_label_vocab(): subclass -> int mapping
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class CacheManifest:
    n_genes: int
    gene_symbol: list[str]
    ensembl_id: list[str] | None
    hvg_indices: list[int]
    L: int
    label_col: str
    label_vocab: dict[str, int]
    qc: dict[str, Any]
    splits: dict[str, int]   # cell counts per split

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(self.__dict__, fh, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "CacheManifest":
        with Path(path).open() as fh:
            payload = json.load(fh)
        return cls(**payload)


# ---------------------------------------------------------------------------
# Build cache (writes)
# ---------------------------------------------------------------------------

def write_cache(
    adata: ad.AnnData,
    split_indices: dict[str, np.ndarray],
    hvg_indices: np.ndarray,
    L: int,
    label_col: str,
    out_dir: Path,
    qc_summary: dict | None = None,
) -> CacheManifest:
    """Materialize the cache directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_vocab = _build_label_vocab(adata, label_col, split_indices.get("train", None))

    splits_sizes: dict[str, int] = {}
    for split, idx in split_indices.items():
        X_split = adata.X[idx]
        if not sparse.issparse(X_split):
            X_split = sparse.csr_matrix(X_split)
        X_split = X_split.tocsr().astype(np.int32)
        sparse.save_npz(out_dir / f"{split}_X.npz", X_split, compressed=True)

        obs_split = adata.obs.iloc[idx].copy()
        # Add integer label column
        obs_split["label"] = obs_split[label_col].astype(str).map(label_vocab).astype("Int64")
        obs_split.to_parquet(out_dir / f"{split}_obs.parquet")
        splits_sizes[split] = int(len(idx))

    manifest = CacheManifest(
        n_genes=int(adata.n_vars),
        gene_symbol=adata.var["gene_symbol"].astype(str).tolist()
        if "gene_symbol" in adata.var.columns
        else adata.var.index.astype(str).tolist(),
        ensembl_id=adata.var["ensembl_id"].astype(str).tolist()
        if "ensembl_id" in adata.var.columns
        else None,
        hvg_indices=hvg_indices.astype(int).tolist(),
        L=int(L),
        label_col=label_col,
        label_vocab=label_vocab,
        qc=qc_summary or {},
        splits=splits_sizes,
    )
    manifest.to_json(out_dir / "manifest.json")
    return manifest


def _build_label_vocab(
    adata: ad.AnnData, label_col: str, train_idx: np.ndarray | None
) -> dict[str, int]:
    if label_col not in adata.obs.columns:
        raise KeyError(f"label_col '{label_col}' not in adata.obs.")
    obs = adata.obs.iloc[train_idx] if train_idx is not None else adata.obs
    labels = sorted(obs[label_col].astype(str).unique())
    return {lbl: i for i, lbl in enumerate(labels)}


# ---------------------------------------------------------------------------
# Read cache (Dataset)
# ---------------------------------------------------------------------------

class CellShardDataset:
    """Random-access dataset over a cache split.

    Not subclassing torch.utils.data.Dataset to avoid importing torch in tests
    that don't need it. A thin adapter is provided in cellfm.training.loop.
    """

    def __init__(self, cache_dir: Path, split: str):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.manifest = CacheManifest.from_json(self.cache_dir / "manifest.json")
        self.X = sparse.load_npz(self.cache_dir / f"{split}_X.npz").tocsr()
        self.obs = pd.read_parquet(self.cache_dir / f"{split}_obs.parquet")
        if "label" not in self.obs.columns:
            raise RuntimeError(
                f"'label' column not in {split}_obs.parquet. Re-run write_cache."
            )
        self.labels = (
            pd.to_numeric(self.obs["label"], errors="coerce")
            .fillna(-100)
            .to_numpy()
            .astype(np.int64)
        )
        if self.X.shape[0] != self.labels.shape[0]:
            raise RuntimeError(
                f"Shape mismatch: X has {self.X.shape[0]} rows, obs has {self.labels.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> dict:
        row = self.X.getrow(idx)
        return {
            "gene_idx": row.indices.astype(np.int64).copy(),
            "values": row.data.astype(np.float32).copy(),
            "label": int(self.labels[idx]),
        }

    @property
    def n_genes(self) -> int:
        return self.manifest.n_genes

    @property
    def hvg_indices(self) -> np.ndarray:
        return np.asarray(self.manifest.hvg_indices, dtype=np.int64)

    @property
    def L(self) -> int:
        return self.manifest.L

    @property
    def n_classes(self) -> int:
        return len(self.manifest.label_vocab)
