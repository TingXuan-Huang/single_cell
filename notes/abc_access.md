# Allen Brain Cell (ABC) Atlas access reference

Quick reference for using `abc_atlas_access` to pull the WMB-10X **isocortex**
slice for this project. Keep this file alongside `notes/pipeline_v1_plan.md`.

> Authoritative docs: <https://github.com/AllenInstitute/abc_atlas_access>  
> WMB-10X overview: <https://alleninstitute.github.io/abc_atlas_access/descriptions/WMB-10X.html>

---

## TL;DR

```bash
# After `pip install -e .[allen]` on the Hyak node:
python -m cellfm.data.download --out /gscratch/<group>/data/raw/abc --list
python -m cellfm.data.download --out /gscratch/<group>/data/raw/abc
```

This downloads:
- All `WMB-10Xv2-Isocortex-*/raw` shards (4 packages)
- All `WMB-10Xv3-Isocortex-*/raw` shards (a handful; count depends on the release)
- `cell_metadata_with_cluster_annotation` and `gene` tables from `WMB-10X`

Total: ~30-50 GB on disk depending on the active release. Don't run this on a laptop.

---

## Manifest layout

`AbcProjectCache.from_s3_cache(download_dir)` exposes the following entities
(properties unless noted):

| Call | Type | Returns |
|---|---|---|
| `abc_cache.list_directories` | property | list of top-level directories in the current manifest |
| `abc_cache.current_manifest` | property | which release/manifest is loaded |
| `abc_cache.list_data_files(directory)` | method | expression-matrix file names in a directory (older releases) |
| `abc_cache.list_expression_matrix_files(directory)` | method | same, newer alias (use as fallback) |
| `abc_cache.list_metadata_files(directory)` | method | metadata file names in a directory |
| `abc_cache.get_data_path(directory, file_name)` | method | downloads the matrix, returns local path |
| `abc_cache.get_metadata_path(directory, file_name)` | method | downloads the metadata CSV, returns local path |
| `abc_cache.get_metadata_dataframe(directory, file_name)` | method | returns the CSV as a pandas DataFrame |

`src/cellfm/data/download.py` wraps these calls and is the only place we
interact with the SDK.

## Expected isocortex packages

A typical 2024-2025 release of WMB-10X enumerates (filtered to isocortex `/raw`):

```
WMB-10Xv2 / WMB-10Xv2-Isocortex-1/raw
WMB-10Xv2 / WMB-10Xv2-Isocortex-2/raw
WMB-10Xv2 / WMB-10Xv2-Isocortex-3/raw
WMB-10Xv2 / WMB-10Xv2-Isocortex-4/raw
WMB-10Xv3 / WMB-10Xv3-Isocortex-1/raw
WMB-10Xv3 / WMB-10Xv3-Isocortex-2/raw
... (more depending on the release)
```

Run `--list` once first; do not bake the file names into a config. The exact
count and naming can change across Allen releases.

## On-disk layout after download

`abc_atlas_access` writes a versioned tree:

```
data/raw/abc/
├── expression_matrices/
│   ├── WMB-10Xv2/<release>/WMB-10Xv2-Isocortex-1-raw.h5ad
│   ├── WMB-10Xv2/<release>/WMB-10Xv2-Isocortex-2-raw.h5ad
│   ├── ...
│   └── WMB-10Xv3/<release>/...
├── metadata/
│   └── WMB-10X/<release>/
│       ├── cell_metadata_with_cluster_annotation.csv
│       └── gene.csv
└── download_manifest.json    # written by our wrapper
```

The `<release>` segment is a date string like `20241115`; the SDK manages it.

## Important cell-metadata columns (downstream pipeline)

From `cell_metadata_with_cluster_annotation.csv`:

| Column | Used for |
|---|---|
| `cell_label` | row index into expression matrix |
| `donor_label` | donor identity for donor-stratified split |
| `subclass` | label for supervised probes / classification head |
| `class` | coarser label (audit only) |
| `cluster` | finer label (audit only) |
| `region_of_interest` | sanity-check that filter == isocortex |
| `library_method` | 10Xv2 vs 10Xv3 (covariate of interest) |

Gene table (`gene.csv`):

| Column | Used for |
|---|---|
| `gene_identifier` (ensembl) | stable join key across h5ads |
| `gene_symbol` | human-readable HVG report |

## Concat strategy

The downloader produces N per-region shards rather than one merged h5ad.
`scripts/build_cache.py` accepts `--input-h5ad-glob` and concatenates the
shards in-memory with `anndata.concat(..., axis=0, join='outer')` before
running QC. The Allen `.var` tables match across shards so the outer join is
a no-op for genes; cell barcodes are made unique with the `index_unique='-'`
suffix.

## Storage budget

| Resource | Approx size |
|---|---|
| WMB-10Xv2 isocortex (4 shards, raw) | ~12-18 GB |
| WMB-10Xv3 isocortex (raw) | ~15-25 GB |
| WMB-10X metadata | < 100 MB |
| Cache after preprocessing | ~3-6 GB (sparse npz) |

Plan ≥ 100 GB of `/gscratch` free for headroom. The full WMB-10Xv2 directory
is ~104 GB so make sure the isocortex filter is active.

## Failure modes worth knowing

1. **S3 throttling / partial download**: `abc_atlas_access` is hash-checked
   and resumable. Just rerun; already-complete files are skipped.
2. **Release drift**: If a new release renames a file you'll see a
   `KeyError` from `get_data_path`. Rerun `--list` to inspect the new
   manifest.
3. **`list_data_files` missing**: The newer SDK versions renamed it to
   `list_expression_matrix_files`. Our wrapper tries both.
4. **Out of disk**: The S3 cache silently fills `download_dir`; check
   `du -sh data/raw/abc/` before starting.
5. **Cell-metadata join**: The expression h5ads carry `obs.index = cell_label`
   so a left-join on `cell_label` against the metadata CSV is the canonical
   wiring step (we do this inside `build_cache.py`).
