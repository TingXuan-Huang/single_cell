# Pipeline v1 Plan: Allen WMB-10X + Small-Model Embedding Bake-off

Status: APPROVED 2026-05-19. Locked decisions below.
Scope: Track B only (Track A deprioritized for now).
Target compute: UW Hyak cluster (SLURM). Local laptop is for unit tests only.

---

## One-sentence goal

Build a clean WMB-10X isocortex data pipeline, train 1M and 3M parameter models
with four different input encoders on the same data, and measure which encoder
preserves biology best.

## Why this scoping is defensible

From `memory/TRACK_B_RESEARCH_PLAN.md` taste decision:
> A smaller model that answers the biological question is stronger than a
> larger model that only improves loss.

This plan tests Hypothesis H2 (rank vs value-aware tokenization, §Core Hypotheses)
at minimum cost, before committing to 10M/30M/100M scaling. The encoder choice
is locked first, scale is locked second.

---

## Locked decisions (2026-05-19)

| # | Decision | Choice |
|---|---|---|
| 1 | Param matching across encoders | **Report params separately, match data + steps.** Embedding table dominates; that's fine. |
| 2 | Train/val/test split | **Donor-stratified, fixed seed = 0.** No donor in two splits. |
| 3 | Value binning scheme | **51 bins per scGPT.** Cell-relative binning of nonzero values. Zero is its own token. |
| 4 | Sequence length L | **95th percentile of nnz, rounded up to power of 2.** Likely 1024 or 2048. Default config: 2048; data probe will lock the real value. |
| 5 | Eval probes | **Both linear and kNN (k=15).** Both rows in the comparison table. |
| 6 | This plan lives at | `notes/pipeline_v1_plan.md` |

---

## Data corpus

- Source: Allen Brain Cell Atlas (ABC), WMB-10X scRNA-seq
- Subset: **mouse isocortex** (smallest viable adult cortex slice; expand later)
- Access: `abc_atlas_access` Python package + AWS S3
- License: CC BY-NC 4.0
- Reference: Yao et al., Nature 2023 (`memory/LEARNINGS.md#wmb-10x-scrnaseq`)
- Expected scale (isocortex only): ~500k-1M cells, ~20k genes, sparse counts

---

## Phase breakdown

### Phase D1: Data pipeline

| Step | Output |
|---|---|
| Download isocortex package via `abc_atlas_access` | `data/raw/<package>.h5ad` (gitignored) |
| AnnData inspection | `notes/abc_isocortex_report.md` |
| QC: mt%, nnz floor, doublet flag | `src/data/qc.py` |
| Donor-stratified train/val/test split (70/15/15, seed=0) | `data/cache/splits.json` |
| HVG selection on train only (top 2k) | `data/cache/hvg_2k.json` |
| nnz histogram + L computation | `notes/figures/nnz_hist.png` |
| Memmap shards for fast loading | `data/cache/{train,val,test}.npz` |

### Phase D2: Encoders (4 interchangeable)

| Encoder | Output shape | Notes |
|---|---|---|
| `hvg_dense` | `(B, 2000)` float32 | Log1p + optional z-score |
| `embedding_bag` | sparse `(indices, offsets, values)` | Weighted nn.EmbeddingBag |
| `rank` | `(B, L)` int + mask | Top-L by expression, Geneformer-style |
| `value_bin` | `(B, L, 2)` int (gene_id, bin_id) | scGPT-style, 51 bins |

### Phase M1: Models

Shared backbone, swappable head:

- `src/models/tiny_transformer.py`: encoder body
- `src/models/heads/{hvg_dense, embedding_bag, gene_token, gene_value_token}.py`
- `src/models/pretraining.py`: masked-token loss + supervised loss

Four size targets (transformer body only; embedding table excluded):

| Size tag | d_model | n_layers | n_heads | ffn_mult | head_dim | body params |
|---|---|---|---|---|---|---|
| `tiny_1m`  | 128 | 2 | 4 | 4 | 32 | ~0.9M  |
| `tiny_3m`  | 192 | 4 | 6 | 4 | 32 | ~2.2M  |
| `tiny_5m`  | 256 | 6 | 8 | 4 | 32 | ~5.2M  |
| `tiny_10m` | 320 | 8 | 8 | 4 | 40 | ~10.5M |

Gene embedding table (~25k × d_model) is reported separately. For HVG-dense and
EmbeddingBag, the head linear layer dominates.

### Phase T1: Training

- `src/training/loop.py`: PyTorch training loop, mixed precision, grad clip
- `src/training/schedules.py`: cosine with warmup
- `scripts/train_one.py`: `--encoder` + `--size` → one wandb run + checkpoint
- 16 runs: 4 encoders × 4 sizes
- Each run target: ~30 min on one A40 GPU (`tiny_10m` drops batch to 64 to fit VRAM)

### Phase E1: Evaluation

Same suite for every run:

- Linear probe: subclass accuracy + macro-F1
- kNN probe (k=15): subclass accuracy + macro-F1
- Geometry: within-class variance trace, participation ratio, kNN-Jaccard vs PCA-64
- Biology: SST/PV/VIP silhouette, OPC cycle correlation with PC1
- PCA-64 baseline: shared reference

Output: one JSONL row per `(encoder, size)`, aggregated into
`notes/encoder_comparison_table.md` + `notes/figures/encoder_comparison.png`.

---

## Experimental matrix

|  | 1M body | 3M body | 5M body | 10M body |
|---|---|---|---|---|
| HVG-dense (supervised)              | run | run | run | run |
| EmbeddingBag (supervised + recon)   | run | run | run | run |
| Rank-token (MLM)                    | run | run | run | run |
| Value-bin token (MLM, 51 bins)      | run | run | run | run |
| PCA-64                              | eval only (shared baseline) |||| 
| scVI (n_latent=32)                  | eval only (shared baseline) ||||

Total: 16 training runs + 1 PCA fit + 1 scVI fit. Whole sweep ~2 GPU-days on Hyak
(parallelizable as a 16-wide array → ~8 wall-clock hours).

---

## Pretraining objectives per encoder

- HVG-dense → supervised subclass classification (no MLM possible without tokens)
- EmbeddingBag → supervised + reconstruction (mask 15% of nonzero gene values)
- Rank-token → masked gene ID prediction (BERT-style, 15% mask)
- Value-bin token → masked joint gene+bin prediction

This intentional asymmetry tests Hypothesis H1 (supervised vs self-supervised
collapse, from `memory/TRACK_B_RESEARCH_PLAN.md`).

---

## File structure

```
LLM_from_scratch/
├── data/                       # gitignored
│   ├── raw/
│   └── cache/
├── memory/                     # existing research docs
├── notes/                      # writeups, reports, figures
├── src/
│   ├── data/                   # download, qc, splits, hvg, cache, synthetic
│   ├── tokenizers/             # 4 input encoders + base interface
│   ├── models/                 # transformer body, heads, pretraining
│   ├── training/               # loop, schedules
│   ├── eval/                   # probes, geometry, biology, pca, compare
│   └── metrics/                # collapse, etc.
├── configs/
│   ├── data/isocortex.yaml
│   ├── models/{tiny_1m, tiny_3m}.yaml
│   ├── encoders/{hvg, embed_bag, rank, value_bin}.yaml
│   └── experiments/<encoder>_<size>.yaml      # 8 combinations
├── scripts/                    # entrypoints (CLI)
├── slurm/                      # Hyak SLURM scripts
├── tests/                      # synthetic-data unit tests
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## UW Hyak deployment notes

- Cluster: `klone` (general access) or `hyak-noodle` depending on account
- Partition: `gpu-a40` for training runs (set in SLURM scripts as placeholder)
- Account: user provides via `--account=` flag (placeholder in SLURM scripts)
- Python: load module `python/3.11` and create venv, or use Apptainer
- Storage: project goes in `/gscratch/<group>/<user>/LLM_from_scratch`
- Data: download once to `/gscratch/.../data/raw`, cache to `/gscratch/.../data/cache`
- Logs: SLURM stdout to `slurm/logs/`, wandb to user account (key in env)

SLURM scripts provided:
- `slurm/download.sbatch` — one-time data download (CPU node, big disk)
- `slurm/preprocess.sbatch` — QC + split + HVG + cache (CPU node)
- `slurm/train_one.sbatch` — single encoder × size run (GPU node)
- `slurm/sweep.sh` — submits all 8 train_one jobs as a job array
- `slurm/eval_one.sbatch` — eval after training (GPU node)

---

## What ships in this commit

- All code (Phases D1-E1)
- All configs (8 experiments + PCA)
- All SLURM scripts (with user-fillable placeholders)
- Unit tests with synthetic AnnData
- README with Hyak quickstart

What does NOT ship:
- Real Allen data downloads (must run on Hyak)
- Training runs (must run on Hyak)
- Evaluation tables (depend on training runs)

---

## Definition of done for this implementation

1. `pytest tests/` passes locally with no GPU and no Allen data
2. `python scripts/train_one.py --encoder rank --size tiny_1m --synthetic` runs end-to-end on synthetic data on laptop
3. SLURM scripts parse-check (`sbatch --test-only`)
4. README has Hyak setup recipe top to bottom

---

## Open questions deferred to runtime

- Exact isocortex package name (depends on current Allen S3 manifest)
- Real L from real nnz distribution (the probe will lock it)
- Subclass label vocabulary size (depends on isocortex slice)
- Mask ratio if val loss plateaus too early (default 15%)

---

## v1.1 amendment (LOCKED 2026-05-19)

User asked: "Add maybe a VAE Encoder as well, and add 5M, 10M model for experiments as well."
Both decisions locked on 2026-05-19: **D1 = B (scVI baseline)** and
**D2 = A (4-point scaling sweep: 1M / 3M / 5M / 10M)**.

### D1 (LOCKED — B) — VAE integration

Decision: **B — scVI as a baseline alongside PCA-64.** Reasons:

- scvi-tools is the field-standard cell VAE (ZINB likelihood, β scheduling,
  library-size factor). Rolling our own VAE inside a 30-min-per-run bake-off
  reinvents it badly.
- `memory/TRACK_B_RESEARCH_PLAN.md` and `memory/LEARNINGS.md` already list
  scVI as a planned baseline under Direction D ("compress then dense"), which
  is architecturally distinct from the Direction-C ("tokenize then sequence
  model") encoders in the bake-off matrix.
- Putting scVI in a baseline slot keeps the bake-off comparison honest
  (apples-to-apples among tokenizer encoders) AND closes the
  Track-B-promised scVI comparison for free.

Plan delta (landed):
- New file: `scripts/build_scvi_baseline.py` (mirrors `build_pca_baseline.py`).
- New file: `slurm/scvi_baseline.sbatch` (mirrors `pca_baseline.sbatch`).
- `"scvi-tools>=1.1"` added to `pyproject.toml` `[allen]` extras.
- Eval consumes the latent `.npz` exactly like pca64; no changes to the
  encoder matrix.
- One new comparison-table row: `scvi` (size column = `zd32` for the chosen latent dim).
- `scripts/build_comparison_table.py --scvi-baseline <dir>` mirrors `--pca-baseline`.

The full-custom-VAE alternative (D1=A) was rejected: scvi-tools already
implements ZINB likelihood + β scheduling + library-size factor correctly,
and rolling our own inside a 30-min-per-run bake-off would reinvent it badly.
The four extra failure modes it would add (KL collapse, posterior collapse,
new factory branch, new eval extraction path) are not worth the duplication.

### D2 (LOCKED — A) — Scaling sweep depth

Decision: **A — add `tiny_5m` and `tiny_10m`** for a 4-point scaling
curve. Reasons:

- 2 points = direction; 4 points = fittable slope.
- At ~10M body params, the 25k × d_model gene-embedding table no longer
  dominates total params for `rank` / `value_bin` — only there does the
  transformer body itself become the load-bearing component, which is the
  whole point of locked decision #1.
- Compute cost: ~8 A40-hours total (16 runs × 30 min); parallelizable on a
  Hyak GPU array. Wall-clock unchanged.
- Connects v1 to Track B's scaling-law ladder (1M / 10M / 30M / 100M);
  without 10M, v1 doesn't reach the bottom rung of v2.

Proposed size configs (body params, transformer body only):

| Size tag    | d_model | n_layers | n_heads | ffn_mult | head_dim | body params |
|-------------|---------|----------|---------|----------|----------|-------------|
| `tiny_1m`   | 128     | 2        | 4       | 4        | 32       | ~0.9M       |
| `tiny_3m`   | 192     | 4        | 6       | 4        | 32       | ~2.2M       |
| `tiny_5m`   | 256     | 6        | 8       | 4        | 32       | ~5.2M (proposed) |
| `tiny_10m`  | 320     | 8        | 8       | 4        | 40       | ~10.5M (proposed) |

Geometric spacing in body params: roughly ×2.5, ×2.4, ×2.0 — close enough
to log-uniform for a slope fit.

Plan delta (landed):
- New files: `configs/model/tiny_5m.yaml`, `configs/model/tiny_10m.yaml`.
- `src/cellfm/models/factory.py` SIZE_CONFIGS: `tiny_5m`, `tiny_10m`
  entries added per the table above.
- `slurm/_runs.sh`: single source of truth for the 16-entry run list
  (RUN_IDS / ENCODERS / SIZES), sourced by both train_array and eval_array
  (also fixes F1 — index drift).
- `slurm/train_array.sbatch` and `slurm/eval_array.sbatch`:
  `--array=0-15`; per-size batch-size policy (`tiny_10m` → 64) lives in
  `_runs.sh::batch_size_for_size`.
- `scripts/build_comparison_table.py`: per-encoder scaling-slope column
  (`slope_acc_per_log2_params`) — linear fit of `linear_acc` vs
  log₂(body_params).

### Architecture review findings (must address regardless of D1/D2)

| # | Finding | Severity | Confidence |
|---|---------|----------|-----------|
| F1 | Slurm `ENCODERS` and `SIZES` arrays duplicated across train/eval scripts; index mismatch silently loads wrong checkpoint. Refactor opportunity, not blocking. | P2 | 8/10 |
| F2 | `build_comparison_table.py` should add per-encoder scaling-slope column to justify the added compute (otherwise you bought 8 runs to print 8 cells). | P2 | 9/10 |
| F3 | **VRAM at `tiny_10m`, L=2048, B=128 ≈ 45 GB on a 48 GB A40 — tight.** Verify `transformer_body.py` uses PyTorch 2 `scaled_dot_product_attention` (flash). If not, drop `batch_size` to 64 for `tiny_10m` only. | P1 | 8/10 |
| F4 | `build_model` knob surface needs `kl_weight` + `kl_warmup_steps` if D1=A. N/A for D1=B or D1=C. | P2 | 9/10 |
| F5 | `tests/test_trainer_smoke.py` only covers `tiny_1m`. Parametrize across all sizes with a SHRUNK preset (override `L=64`, `batch=4`) so the test stays CPU-runnable in ~5s. **Critical test gap** if sizes land unparametrized. | P1 | 9/10 |
| F6 | If D1=B, add a synthetic-data scVI baseline test mirroring the pca64 test. ~30 lines. | P2 | 8/10 |
| F7 | Step count `n_steps=30000` is constant across sizes. "Match steps" ≠ "match compute" across 10× param count. Locked Open #1 ("match data + steps") implicitly accepts this; one-line acknowledgement in the comparison table writeup is enough. | P3 | 9/10 |

### Critical failure mode

`tiny_10m` OOM at L=2048 + B=128 + non-flash attention. Silent because the
SLURM array continues with other sizes, the eval array later loads whatever
checkpoint (if any) exists, and the comparison table prints NaN. **Mitigation
to add to the plan if D2=A approved:**

1. Add a 1-step VRAM probe at the start of `train_one.py` for `tiny_10m`:
   forward + backward on a single batch, log peak memory, abort with a clear
   message (not OOM) if it exceeds 90% of available VRAM.
2. Set `batch_size=64` as the explicit `tiny_10m` override in
   `configs/experiment/main_grid.yaml`.
3. Assert flash attention is engaged in `transformer_body.py` (PyTorch 2
   SDPA backend selector).

### NOT in scope (this amendment)

- Custom from-scratch VAE encoder (deferred; scVI covers the role).
- Sizes >10M (Track B v2: 30M, 100M).
- ZINB likelihood for our transformer encoders (scVI handles it).
- Eval-time latent-space interpolation / arithmetic studies.
- Re-tuning step count per size (Open #1 stands: match steps).

### What already exists

- `models/factory.py` SIZE_CONFIGS → trivial to add tiny_5m/tiny_10m.
- `slurm/pca_baseline.sbatch` + `scripts/build_pca_baseline.py` → exact
  template for scVI baseline.
- `cellfm.eval.probes` → consumes any (N, d) embedding; scVI latents fit.
- `build_comparison_table.py` iterates run dirs → wider grid is data-driven.

### Resolved decisions (2026-05-19)

- **D1 — VAE framing:** B (scVI baseline). Done — `scripts/build_scvi_baseline.py`,
  `slurm/scvi_baseline.sbatch`, `[allen]` extra, comparison-table row landed.
- **D2 — Scaling sweep depth:** A (1M / 3M / 5M / 10M, 16 runs). Done — size
  configs, factory entries, slurm array bumped to `0-15` via `_runs.sh`.

## GSTACK REVIEW REPORT

| Review        | Trigger              | Why                              | Runs | Status               | Findings                                                            |
|---------------|----------------------|----------------------------------|------|----------------------|---------------------------------------------------------------------|
| CEO Review    | `/plan-ceo-review`   | Scope & strategy                 | 0    | —                    | —                                                                   |
| Eng Review    | `/plan-eng-review`   | Architecture & tests (required)  | 1    | CLEARED              | All findings addressed (see below)                                  |
| Design Review | `/plan-design-review`| UI/UX gaps                       | 0    | —                    | —                                                                   |
| Adversarial   | `/codex review`      | Independent 2nd opinion          | 0    | —                    | —                                                                   |
| Outside Voice | `/codex consult`     | Cross-model challenge            | 0    | —                    | —                                                                   |

Finding resolutions:
- F1 (P2, index drift): RESOLVED — single source of truth in `slurm/_runs.sh`.
- F2 (P2, scaling-slope column): RESOLVED — `slope_acc_per_log2_params` added.
- F3 (P1, tiny_10m VRAM): RESOLVED — `batch_size_for_size` drops `tiny_10m` to 64;
  SDPA fast path verified in `transformer_body.py` (head_dim=40 stays in fast tier).
- F4 (P2, KL knobs): N/A — D1=B selected; no custom VAE.
- F5 (P1, parametrized smoke test): RESOLVED — `tests/test_trainer_smoke.py`
  parametrizes 4 encoders × 4 sizes with the SHRUNK synthetic cache.
- F6 (P2, synthetic scVI test): RESOLVED — `tests/test_scvi_baseline.py`
  guarded by `pytest.importorskip("scvi")`.
- F7 (P3, match-steps caveat): Accepted; comparison-table writeup will note it.

- **VERDICT:** CLEARED (2026-05-19). D1=B and D2=A locked; all P1/P2 findings
  addressed. Ready for end-to-end Hyak runs.
