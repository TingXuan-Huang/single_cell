#!/usr/bin/env bash
#SBATCH --job-name=cellfm-eval
#SBATCH --partition=gpu-a40
#SBATCH --account=PI_LAB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-15
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

set -euo pipefail
: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-$(pwd)}}"
SLURM_DIR="$PROJECT_ROOT/slurm"
source "$SLURM_DIR/_common.sh"
source "$SLURM_DIR/_runs.sh"
cd "$PROJECT_ROOT"

CACHE_DIR="${CACHE_DIR:-/gscratch/${SLURM_JOB_ACCOUNT}/data/cache/wmb_isocortex_v1}"
OUT_ROOT="${OUT_ROOT:-/gscratch/${SLURM_JOB_ACCOUNT}/runs/cellfm/v1}"
PCA_REF="${PCA_REF:-$OUT_ROOT/pca64/embeddings_test.npz}"

idx=$SLURM_ARRAY_TASK_ID
if [[ $idx -ge $N_RUNS ]]; then
  echo "[hyak] array index $idx >= N_RUNS=$N_RUNS; nothing to do" >&2
  exit 0
fi
ENCODER=${ENCODERS[$idx]}
SIZE=${SIZES[$idx]}
RUN_ID=${RUN_IDS[$idx]}
RUN_DIR="$OUT_ROOT/$RUN_ID"

python -m scripts.eval_one \
  --run-dir "$RUN_DIR" \
  --cache-dir "$CACHE_DIR" \
  --encoder-config "configs/encoder/${ENCODER}.yaml" \
  --model-config   "configs/model/${SIZE}.yaml" \
  --pca-ref "$PCA_REF" \
  --num-workers 4
