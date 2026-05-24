#!/usr/bin/env bash
# Submit cache build first, then launch training arrays after the cache succeeds.
#
# This is a submitter script, not an sbatch payload. Run it from the repo root
# on Hyak with: bash slurm/cache_then_train_hyak.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-stf}"
PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/stf/thuang27/single_cell}"
VENV_DIR="${VENV_DIR:-/gscratch/stf/thuang27/envs/llm}"
DATA_CFG="${DATA_CFG:-configs/data/wmb_isocortex_hyak.yaml}"
CACHE_DIR="${CACHE_DIR:-/gscratch/stf/thuang27/data/cache/wmb_isocortex_v1}"
OUT_ROOT="${OUT_ROOT:-/gscratch/stf/thuang27/runs/cellfm/v1}"
WANDB_PROJECT="${WANDB_PROJECT:-cellfm-v1}"

CACHE_PARTITION="${CACHE_PARTITION:-cpu-g2-mem2x}"
CACHE_MEM="${CACHE_MEM:-384G}"
CACHE_CPUS="${CACHE_CPUS:-16}"

STABLE_GPU_PARTITION="${STABLE_GPU_PARTITION:-gpu-2080ti}"
STABLE_GPU_GRES="${STABLE_GPU_GRES:-gpu:1}"
STABLE_ARRAY="${STABLE_ARRAY:-0-3%4}"

CKPT_PARTITION="${CKPT_PARTITION:-ckpt-all}"
CKPT_GPU_GRES="${CKPT_GPU_GRES:-gpu:a40:1}"
CKPT_ARRAY="${CKPT_ARRAY:-4-15%4}"

cd "$PROJECT_ROOT"
mkdir -p logs "$OUT_ROOT"

COMMON_EXPORT="ALL,PROJECT_ROOT=$PROJECT_ROOT,VENV_DIR=$VENV_DIR,WANDB_PROJECT=$WANDB_PROJECT,PYTHONNOUSERSITE=1"

J_CACHE=$(
  sbatch --parsable \
    --job-name=cellfm-cache \
    --account="$ACCOUNT" \
    --partition="$CACHE_PARTITION" \
    --mem="$CACHE_MEM" \
    --cpus-per-task="$CACHE_CPUS" \
    --export="$COMMON_EXPORT,DATA_CFG=$DATA_CFG" \
    slurm/build_cache.sbatch
)
echo "[hyak] cache -> $J_CACHE"

J_STABLE=$(
  sbatch --parsable \
    --dependency=afterok:"$J_CACHE" \
    --job-name=cellfm-1m-2080ti \
    --account="$ACCOUNT" \
    --partition="$STABLE_GPU_PARTITION" \
    --gres="$STABLE_GPU_GRES" \
    --array="$STABLE_ARRAY" \
    --export="$COMMON_EXPORT,CACHE_DIR=$CACHE_DIR,OUT_ROOT=$OUT_ROOT" \
    slurm/train_array.sbatch
)
echo "[hyak] stable train -> $J_STABLE (after $J_CACHE, array $STABLE_ARRAY)"

J_CKPT=$(
  sbatch --parsable \
    --dependency=afterok:"$J_CACHE" \
    --job-name=cellfm-rest-ckpt \
    --account="$ACCOUNT" \
    --partition="$CKPT_PARTITION" \
    --gres="$CKPT_GPU_GRES" \
    --array="$CKPT_ARRAY" \
    --export="$COMMON_EXPORT,CACHE_DIR=$CACHE_DIR,OUT_ROOT=$OUT_ROOT" \
    slurm/train_array.sbatch
)
echo "[hyak] checkpoint train -> $J_CKPT (after $J_CACHE, array $CKPT_ARRAY)"

echo "[hyak] monitor: squeue -u $USER -o \"%.18i %.24j %.12P %.8T %.10M %.10l %R\""
