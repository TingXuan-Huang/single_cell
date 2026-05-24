from __future__ import annotations

import shutil

import pandas as pd

from cellfm.data.cache import CacheManifest, CellShardDataset


def test_cache_manifest_loads(synthetic_cache):
    m = CacheManifest.from_json(synthetic_cache / "manifest.json")
    assert m.n_genes > 0
    assert m.L >= 64
    assert len(m.hvg_indices) > 0
    assert "train" in m.splits and "val" in m.splits and "test" in m.splits


def test_cell_shard_dataset_getitem(synthetic_cache):
    ds = CellShardDataset(synthetic_cache, split="train")
    assert len(ds) > 0
    item = ds[0]
    assert set(item) >= {"gene_idx", "values", "label"}
    assert item["gene_idx"].dtype.kind in ("i", "u")
    assert item["values"].dtype == "float32"
    assert isinstance(item["label"], int)
    # Nonzero genes only
    assert item["gene_idx"].size == item["values"].size


def test_cell_shard_dataset_converts_nullable_labels_to_ignore_index(synthetic_cache, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for name in ("manifest.json", "train_X.npz"):
        shutil.copy(synthetic_cache / name, cache_dir / name)
    obs = pd.read_parquet(synthetic_cache / "train_obs.parquet")
    obs.loc[obs.index[0], "label"] = pd.NA
    obs.to_parquet(cache_dir / "train_obs.parquet")

    ds = CellShardDataset(cache_dir, split="train")

    assert ds.labels[0] == -100
