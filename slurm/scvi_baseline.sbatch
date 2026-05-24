#!/usr/bin/env bash
#SBATCH --job-name=cellfm-scvi32
#SBATCH --partition=gpu-a40
#SBATCH --account=PI_LAB
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/scvi_%j.out
#SBATCH --error=logs/scvi_%j.err

# scVI baseline. Lives in `baselines:` alongside PCA-64 (see
# configs/experiment/main_grid.yaml). GPU is optional but ~10x faster on A40.

set -euo pipefail
: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-$(pwd)}}"
SLURM_DIR="$PROJECT_ROOT/slurm"
source "$SLURM_DIR/_common.sh"
cd "$PROJECT_ROOT"

CACHE_DIR="${CACHE_DIR:-/gscratch/${SLURM_JOB_ACCOUNT}/data/cache/wmb_isocortex_v1}"
OUT_DIR="${OUT_DIR:-/gscratch/${SLURM_JOB_ACCOUNT}/runs/cellfm/v1/scvi32}"
N_LATENT="${SCVI_N_LATENT:-32}"
MAX_EPOCHS="${SCVI_MAX_EPOCHS:-200}"
BATCH_KEY="${SCVI_BATCH_KEY:-donor_id}"

python -m scripts.build_scvi_baseline \
  --cache-dir "$CACHE_DIR" \
  --out-dir   "$OUT_DIR" \
  --n-latent "$N_LATENT" \
  --max-epochs "$MAX_EPOCHS" \
  --batch-key "$BATCH_KEY"
