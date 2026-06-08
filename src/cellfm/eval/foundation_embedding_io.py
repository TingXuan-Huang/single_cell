"""I/O helpers for pretrained foundation-model embedding checks."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from cellfm.data.cache import CacheManifest


@dataclass(frozen=True)
class SplitPayload:
    """The sampled cache split used for external extraction and diagnostics."""

    cache_dir: Path
    split: str
    indices: np.ndarray
    X: sparse.csr_matrix
    obs: pd.DataFrame
    y: np.ndarray
    label_text: np.ndarray
    gene_symbol: list[str]
    ensembl_id: list[str] | None


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_named_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=PATH, got {spec!r}. Example: pca64=/path/embeddings_test.npz"
        )
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"Empty embedding name in {spec!r}")
    return name, Path(raw_path).expanduser()


def safe_unique_names(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out = []
    for raw in names:
        name = str(raw) if str(raw) else "unknown_gene"
        count = counts.get(name, 0)
        counts[name] = count + 1
        out.append(name if count == 0 else f"{name}-{count}")
    return out


def load_cache_split(
    cache_dir: Path,
    split: str,
    *,
    max_cells: int | None,
    seed: int,
) -> SplitPayload:
    cache_dir = Path(cache_dir)
    manifest = CacheManifest.from_json(cache_dir / "manifest.json")
    X = sparse.load_npz(cache_dir / f"{split}_X.npz").tocsr()
    obs = pd.read_parquet(cache_dir / f"{split}_obs.parquet").copy()
    y = obs["label"].to_numpy().astype(np.int64)

    indices = np.arange(X.shape[0])
    if max_cells is not None and max_cells > 0 and X.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_cells, replace=False))
        X = X[indices]
        obs = obs.iloc[indices].copy()
        y = y[indices]

    inverse_vocab = {int(v): k for k, v in manifest.label_vocab.items()}
    label_text = np.asarray([inverse_vocab.get(int(i), str(i)) for i in y], dtype=object)
    obs["cellfm_label_id"] = y
    obs["cellfm_label"] = label_text

    return SplitPayload(
        cache_dir=cache_dir,
        split=split,
        indices=indices.astype(np.int64),
        X=X,
        obs=obs,
        y=y,
        label_text=label_text,
        gene_symbol=list(manifest.gene_symbol),
        ensembl_id=list(manifest.ensembl_id) if manifest.ensembl_id is not None else None,
    )


def payload_to_anndata(payload: SplitPayload):
    import anndata as ad

    var = pd.DataFrame(index=safe_unique_names(payload.gene_symbol))
    var["gene_symbol"] = payload.gene_symbol
    if payload.ensembl_id is not None:
        var["ensembl_id"] = payload.ensembl_id

    obs = payload.obs.copy()
    obs["n_counts"] = np.asarray(payload.X.sum(axis=1)).ravel().astype(np.float32)
    obs["filter_pass"] = 1
    obs.index = obs.index.astype(str)

    return ad.AnnData(X=payload.X.copy(), obs=obs, var=var)


def load_embedding_matrix(path: Path, *, h5ad_obsm_key: str) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        preferred = ("X", "X_test", "embeddings", "cell_embeddings", "X_scGPT", "X_uce")
        for key in preferred:
            if key in data and np.asarray(data[key]).ndim == 2:
                return np.asarray(data[key], dtype=np.float32)
        for key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim == 2:
                return arr.astype(np.float32)
        raise ValueError(f"No 2D embedding array found in {path}")
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D .npy embedding matrix, got shape {arr.shape}")
        return arr.astype(np.float32)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        arr = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] == 0:
            raise ValueError(f"No numeric embedding columns found in {path}")
        return arr
    if suffix in {".pt", ".pth"}:
        import torch

        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        elif isinstance(obj, dict):
            arr = None
            for key in ("X", "embeddings", "cell_embeddings"):
                value = obj.get(key)
                if isinstance(value, torch.Tensor) and value.ndim == 2:
                    arr = value.detach().cpu().numpy()
                    break
            if arr is None:
                raise ValueError(f"No 2D tensor under X/embeddings/cell_embeddings in {path}")
        else:
            raise ValueError(f"Unsupported torch object in {path}: {type(obj)!r}")
        return np.asarray(arr, dtype=np.float32)
    if suffix == ".h5ad":
        import anndata as ad

        adata = ad.read_h5ad(path)
        if h5ad_obsm_key not in adata.obsm:
            raise KeyError(f"{path} has no .obsm[{h5ad_obsm_key!r}]")
        return np.asarray(adata.obsm[h5ad_obsm_key], dtype=np.float32)
    raise ValueError(f"Unsupported embedding file type: {path}")


def save_embedding_npz(out_dir: Path, name: str, X: np.ndarray, payload: SplitPayload) -> Path:
    model_dir = out_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "embeddings.npz"
    np.savez_compressed(
        out_path,
        X=np.asarray(X, dtype=np.float32),
        y=payload.y.astype(np.int64),
        label_text=payload.label_text.astype(str),
        sampled_indices=payload.indices.astype(np.int64),
    )
    return out_path
