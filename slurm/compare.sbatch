#!/usr/bin/env bash
#SBATCH --job-name=cellfm-compare
#SBATCH --partition=compute
#SBATCH --account=PI_LAB
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err

set -euo pipefail
: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-$(pwd)}}"
SLURM_DIR="$PROJECT_ROOT/slurm"
source "$SLURM_DIR/_common.sh"
cd "$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-/gscratch/${SLURM_JOB_ACCOUNT}/runs/cellfm/v1}"

python -m scripts.build_comparison_table \
  --run-roots "$OUT_ROOT" \
  --pca-baseline "$OUT_ROOT/pca64" \
  --scvi-baseline "$OUT_ROOT/scvi32" \
  --out "$OUT_ROOT/COMPARISON.md"
