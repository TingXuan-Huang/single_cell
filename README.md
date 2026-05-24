# cellfm

From-scratch brain-specialized cell foundation model on the Allen ABC Atlas.
First milestone: a controlled 1M / 3M parameter encoder bake-off on WMB-10X
isocortex.

See `notes/pipeline_v1_plan.md` for the locked plan.
See `memory/` for the research design, learnings, and Track B research plan.

---

## What this repo does

Trains four small models with four different input encoders on the same Allen
mouse isocortex scRNA-seq data and measures which encoder preserves biology
best.

Encoders compared:
1. `hvg_dense` — top 2k highly variable genes + dense linear (supervised baseline)
2. `embedding_bag` — weighted `nn.EmbeddingBag` (sparse linear; non-transformer baseline)
3. `rank` — top-L gene-ID tokens (Geneformer-style)
4. `value_bin` — gene-ID + binned-expression tokens, 51 bins (scGPT-style)

Each encoder is trained at two parameter scales (`tiny_1m`, `tiny_3m`) and
evaluated on the same suite (linear probe, kNN probe, geometry, biology).
Comparison shipped as `notes/encoder_comparison_table.md`.

---

## Quickstart (local, synthetic data)

For developing on a laptop with no Allen data and no GPU:

```bash
git clone <repo> && cd LLM_from_scratch
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run all unit tests (uses synthetic AnnData fixtures)
pytest

# End-to-end smoke run on synthetic data
python scripts/train_one.py --encoder rank --size tiny_1m --synthetic --steps 50
```

---

## UW Hyak setup

The full step-by-step recipe lives in [`notes/HYAK_RUNBOOK.md`](notes/HYAK_RUNBOOK.md).
TL;DR:

```bash
# Clone, set up venv, install (see runbook for full module-load order)
git clone <repo> LLM_from_scratch && cd LLM_from_scratch
module load python/3.11.4 cuda/12.4
python -m venv .venv && source .venv/bin/activate
pip install -e ".[allen]" wandb pytest

# One-time: replace PI_LAB with your allocation account
sed -i "s/PI_LAB/<your-account>/g" slurm/*.sbatch
mkdir -p logs

# End-to-end (chained via afterok dependencies)
bash slurm/sweep.sh
```

That submits, in order: `download → build_cache → pca_baseline + train_array
→ eval_array → compare`. Outputs land under `/gscratch/<group>/data/cache/`
and `/gscratch/<group>/results/`. See the runbook for storage layout,
config edits, and common gotchas.

---

## Repository layout

```
src/cellfm/
├── data/        download, qc, splits, hvg, cache, synthetic generator
├── tokenizers/  4 input encoders (base interface in base.py)
├── models/      transformer body + 4 input heads + pretraining heads
├── training/    training loop, lr schedules
├── eval/        embedding extraction, probes, geometry, biology, pca baseline
└── metrics/     collapse metrics

configs/
├── data/        isocortex.yaml
├── models/      tiny_1m.yaml, tiny_3m.yaml
├── encoders/    hvg.yaml, embed_bag.yaml, rank.yaml, value_bin.yaml
└── experiments/ 8 (encoder × size) combinations

scripts/         CLI entrypoints
slurm/           SLURM submission scripts for Hyak
tests/           pytest with synthetic AnnData fixtures
notes/           research notes, reports, figures
memory/          long-lived research design and learnings
```

---

## Project status

| Phase | Status |
|---|---|
| D1: Data pipeline | code shipped; awaits Hyak run |
| D2: Encoders | code shipped; unit-tested |
| M1: Models | code shipped; unit-tested |
| T1: Training | code shipped; awaits Hyak run |
| E1: Evaluation | code shipped; awaits checkpoints |

See `notes/pipeline_v1_plan.md` for the full plan and decision log.

---

## Citing / license

Allen data: CC BY-NC 4.0. See `memory/LEARNINGS.md` for dataset references
(Yao et al., Nature 2023).

Code: MIT.
