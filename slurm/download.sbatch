#!/usr/bin/env bash
#SBATCH --job-name=cellfm-download
#SBATCH --partition=compute
#SBATCH --account=PI_LAB
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

# Edit --account=PI_LAB to your Hyak allocation.

set -euo pipefail
: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-$(pwd)}}"
SLURM_DIR="$PROJECT_ROOT/slurm"
source "$SLURM_DIR/_common.sh"
cd "$PROJECT_ROOT"

OUT_DIR="${OUT_DIR:-/gscratch/${SLURM_JOB_ACCOUNT}/data/raw/abc}"
mkdir -p "$OUT_DIR"

# First inspect available isocortex packages, then download all of them.
python -m cellfm.data.download --out "$OUT_DIR" --list
python -m cellfm.data.download --out "$OUT_DIR"
echo "[hyak] download done -> $OUT_DIR"
