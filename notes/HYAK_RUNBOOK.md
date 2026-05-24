# UW Hyak runbook — cellfm pipeline v1

Step-by-step recipe for running the full WMB-10X isocortex pipeline on
**UW Hyak** (`klone`). Companion to `notes/pipeline_v1_plan.md` and
`notes/abc_access.md`.

> All compute must happen on Hyak. Local laptop is for unit tests only.

---

## 0. Prerequisites

You should have:

- A Hyak allocation (an `--account=<group>`).
- Access to one of the GPU partitions, ideally `gpu-a40` or `gpu-l40s`.
- ~150 GB of free space on `/gscratch/<group>`.
- A `wandb` API key (optional but recommended for live logging).

Conventions used below — substitute your own values:

| Placeholder | Replace with |
|---|---|
| `PI_LAB` | your allocation account name |
| `gpu-a40` | your GPU partition |
| `<group>` | your `/gscratch` group |
| `<user>` | your Hyak username |

---

## 1. One-time setup

```bash
ssh klone.hyak.uw.edu

# Land in scratch — never in $HOME.
cd /gscratch/<group>
mkdir -p <user> && cd <user>

# Clone the repo.
git clone <this repo>  LLM_from_scratch
cd LLM_from_scratch

# Module + venv. Hyak provides python/3.11 + cuda modules.
module load python/3.11.4 cuda/12.4

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -e ".[allen]"
pip install wandb pytest

# Quick smoke test (CPU only, ~30 s).
pytest tests/ -q
```

`pytest tests/` should pass with no GPU and no Allen data; it generates a
synthetic AnnData and runs each tokenizer + model forward/backward.

---

## 2. Configure the SLURM scripts

Every `slurm/*.sbatch` script has an `#SBATCH --account=PI_LAB` placeholder.
Edit them once:

```bash
# Replace PI_LAB with your account everywhere:
sed -i "s/PI_LAB/<your-account>/g" slurm/*.sbatch

# Make sure logs/ exists (SLURM --output writes here).
mkdir -p logs

# Optional: edit slurm/_common.sh if your venv lives elsewhere.
```

If your wandb key is in `~/.netrc` or `~/.config/wandb/` it'll be picked up
automatically; otherwise `export WANDB_API_KEY=...` in your `~/.bashrc`.

---

## 3. Download the Allen WMB-10X isocortex slice

```bash
# CPU node, ~6 hours wall clock at peak S3 throughput.
sbatch slurm/download.sbatch
```

Watch progress:

```bash
squeue -u <user>
tail -f logs/download_*.out
```

When the job finishes, verify the manifest:

```bash
cat /gscratch/<group>/data/raw/abc/download_manifest.json | python -m json.tool | head -30
du -sh /gscratch/<group>/data/raw/abc/
# Expect 30-50 GB.
```

If you want to inspect what's available first without downloading:

```bash
srun -p compute -A <your-account> -t 0:30:00 --pty bash
source .venv/bin/activate
python -m cellfm.data.download --out /gscratch/<group>/data/raw/abc --list
```

---

## 4. Edit the data config paths

Open `configs/data/wmb_isocortex.yaml` and update:

- `input_h5ad_glob` — should already point to
  `/gscratch/PROJECT_GROUP/data/raw/abc/expression_matrices/WMB-10X*/*/WMB-10X*-Isocortex-*-raw.h5ad`;
  replace `PROJECT_GROUP` with your group.
- `metadata_csv` — substitute the release date that
  `abc_atlas_access` actually wrote (e.g. `20241115`). Check with:

  ```bash
  ls /gscratch/<group>/data/raw/abc/metadata/WMB-10X/
  ```

- `cache_dir` — replace `PROJECT_GROUP` with your group.

---

## 5. Build the preprocessed cache

```bash
sbatch slurm/build_cache.sbatch
```

This runs on a single CPU node (~64 GB RAM, ~2 hours):

1. Concatenates all isocortex shards via `anndata.concat`.
2. Attaches the cell metadata CSV (donor_label, subclass, ...).
3. Applies QC: nnz floor, mt% ceiling, optional doublet cap.
4. Donor-stratified train/val/test split (70/15/15, seed=0).
5. Picks `L` from the 95th percentile of nonzero counts on train, rounded
   up to the next power of two (clamped to [1024, 2048]).
6. Selects 2000 HVGs on the **train split only**.
7. Writes the sparse-NPZ + parquet cache + `manifest.json` to `cache_dir`.

When it finishes:

```bash
cat /gscratch/<group>/data/cache/wmb_isocortex_v1/manifest.json | python -m json.tool | head -30
cat /gscratch/<group>/data/cache/wmb_isocortex_v1/BUILD_AUDIT.json | python -m json.tool
```

The audit doc shows nnz stats, split sizes, QC drops, and the chosen L.

---

## 6. Fit the PCA-64 baseline (CPU)

```bash
sbatch slurm/pca_baseline.sbatch
# Output: $OUT_ROOT/pca64/{eval_summary.json, embeddings_test.npz}
```

This is the shared reference for all encoder runs.

---

## 6b. Fit the scVI baseline (GPU, optional but recommended)

```bash
sbatch slurm/scvi_baseline.sbatch
# Output: $OUT_ROOT/scvi32/{eval_summary.json, embeddings_test.npz, scvi_model/}
```

scVI sits in the `baselines:` slot alongside PCA-64 (see
`configs/experiment/main_grid.yaml`). It's a "compress then dense" comparison
point distinct from the four Direction-C tokenizer encoders. Override via env:
`SCVI_N_LATENT`, `SCVI_MAX_EPOCHS`, `SCVI_BATCH_KEY`.

---

## 7. Train the encoder × size bake-off (16 GPU jobs)

```bash
sbatch slurm/train_array.sbatch
# (encoder, size) mapping is in slurm/_runs.sh — single source of truth,
# shared with eval_array.sbatch so indices never drift.
```

`slurm/train_array.sbatch` is a SLURM array job (`--array=0-15`, 4 encoders ×
4 sizes). Each task trains one (encoder, size) combination for `n_steps`
(default 5000 ≈ 30 min on a single A40). `tiny_10m` drops batch size to 64
via `batch_size_for_size` in `_runs.sh` to fit a 48 GB A40 at L=2048.

Monitor:

```bash
squeue -u <user>
tail -f logs/train_<jobid>_<index>.out
```

Per-run outputs land under
`$OUT_ROOT/<encoder>_<size>/` (defaults to
`/gscratch/<group>/runs/cellfm/v1/<encoder>_<size>/`):

```
final.pt
best.pt
train_history.json
train_config.json
```

---

## 8. Evaluate (16 GPU jobs)

```bash
sbatch --dependency=afterok:<train_jobid> slurm/eval_array.sbatch
```

Per-run outputs: `$OUT_ROOT/<encoder>_<size>/eval_summary.json` containing
linear-probe, kNN-probe, geometry, neural-collapse, and biology metrics. Uses
the same `_runs.sh` source of truth as the train array so indices line up.

---

## 9. Compare runs

```bash
sbatch --dependency=afterok:<eval_jobid> slurm/compare.sbatch
# Output: $OUT_ROOT/COMPARISON.md (plus matching .csv)
```

`scripts/build_comparison_table.py` accepts both `--pca-baseline` and
`--scvi-baseline` as parallel rows, and emits a per-encoder
`slope_acc_per_log2_params` column from a log₂(body_params) regression — the
load-bearing number for the 4-point scaling sweep.

Copy that file back to `notes/encoder_comparison_table.md` in the repo when
you're ready to commit the result.

---

## 10. End-to-end (recommended)

`slurm/sweep.sh` chains all of the above with `afterok` dependencies:

```bash
bash slurm/sweep.sh
```

It submits download → build_cache → (pca_baseline + scvi_baseline +
train_array) → eval_array → compare, returns the array of job IDs, and exits.
Walk away.

---

## Storage layout reference

```
/gscratch/<group>/
├── <user>/LLM_from_scratch/        # code (git clone)
│   ├── .venv/                       # pip-managed
│   ├── logs/                        # SLURM stdout/stderr
│   └── notes/                       # writeups
├── data/
│   ├── raw/abc/                     # Allen download (~30-50 GB)
│   └── cache/wmb_isocortex_v1/      # preprocessed cache (~3-6 GB)
└── runs/cellfm/v1/                  # checkpoints + eval JSON
    ├── pca64/                       # PCA-64 baseline (embeddings_test.npz + eval_summary.json)
    ├── scvi32/                      # scVI baseline (latent dim 32; same outputs + scvi_model/)
    ├── hvg_dense_tiny_1m/           # 4 encoders × 4 sizes = 16 run dirs
    ├── hvg_dense_tiny_3m/
    ├── hvg_dense_tiny_5m/
    ├── hvg_dense_tiny_10m/
    ├── embedding_bag_tiny_1m/
    ├── ...
    ├── value_bin_tiny_10m/
    ├── COMPARISON.md                # produced by slurm/compare.sbatch
    └── COMPARISON.csv               # ditto, machine-readable
```

`logs/` is checked into the repo as `logs/.gitkeep` so SLURM `--output` paths
resolve before you mkdir anything.

---

## Quick checks before submitting

| Check | Command |
|---|---|
| Account is real | `sacctmgr show user <user>` |
| GPU partition exists | `sinfo -p gpu-a40` |
| Job submits | `sbatch --test-only slurm/train_array.sbatch` |
| Cache is built | `ls /gscratch/<group>/data/cache/wmb_isocortex_v1/manifest.json` |
| wandb works | `python -c "import wandb; wandb.login()"` |

---

## Common gotchas

1. **`module load` order matters**: CUDA must load before `pip install torch`
   or you'll get a CPU-only wheel. We use `--index-url` already in
   `requirements.txt` to avoid this, but verify with
   `python -c "import torch; print(torch.cuda.is_available())"` after install.

2. **`/gscratch` quotas**: Run `gscratch-usage <group>` to check. The pipeline
   needs ~80 GB peak between raw data + cache + checkpoints.

3. **SLURM time limits**: The provided defaults (24h for training,
   6h for download) are generous. The bake-off should finish in <8h end to end.

4. **Crash mid-run**: SLURM checkpoints land in `out_dir`; rerun the same
   `train_one.py` invocation and it will overwrite the checkpoint at the next
   eval boundary.

5. **`wandb` is offline**: Set `WANDB_MODE=offline` if Hyak's outbound HTTP
   is restricted; you can `wandb sync` later.
