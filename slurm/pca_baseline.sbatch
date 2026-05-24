#!/usr/bin/env bash
#SBATCH --job-name=cellfm-pca64
#SBATCH --partition=compute
#SBATCH --account=PI_LAB
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/pca_%j.out
#SBATCH --error=logs/pca_%j.err

set -euo pipefail
: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-$(pwd)}}"
SLURM_DIR="$PROJECT_ROOT/slurm"
source "$SLURM_DIR/_common.sh"
cd "$PROJECT_ROOT"

CACHE_DIR="${CACHE_DIR:-/gscratch/${SLURM_JOB_ACCOUNT}/data/cache/wmb_isocortex_v1}"
OUT_DIR="${OUT_DIR:-/gscratch/${SLURM_JOB_ACCOUNT}/runs/cellfm/v1/pca64}"

python -m scripts.build_pca_baseline \
  --cache-dir "$CACHE_DIR" \
  --out-dir   "$OUT_DIR" \
  --n-components 64
