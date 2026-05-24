#!/usr/bin/env bash
#SBATCH --job-name=cellfm-cache
#SBATCH --partition=compute
#SBATCH --account=PI_LAB
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/cache_%j.out
#SBATCH --error=logs/cache_%j.err

# Preprocess the downloaded H5AD -> sparse cache shards.

set -euo pipefail
: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-$(pwd)}}"
SLURM_DIR="$PROJECT_ROOT/slurm"
source "$SLURM_DIR/_common.sh"
cd "$PROJECT_ROOT"

DATA_CFG="${DATA_CFG:-configs/data/wmb_isocortex.yaml}"

python -m scripts.build_cache --data-config "$DATA_CFG"
echo "[hyak] cache built."
