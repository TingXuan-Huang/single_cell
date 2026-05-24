#!/usr/bin/env bash
#SBATCH --job-name=cellfm-train
#SBATCH --partition=gpu-a40
#SBATCH --account=PI_LAB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-15
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

# Array job: 4 encoders x 4 sizes = 16 runs. The (encoder, size) order is
# defined in slurm/_runs.sh and mirrored by configs/experiment/main_grid.yaml.

set -euo pipefail
: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-$(pwd)}}"
SLURM_DIR="$PROJECT_ROOT/slurm"
source "$SLURM_DIR/_common.sh"
source "$SLURM_DIR/_runs.sh"
cd "$PROJECT_ROOT"

CACHE_DIR="${CACHE_DIR:-/gscratch/${SLURM_JOB_ACCOUNT}/data/cache/wmb_isocortex_v1}"
OUT_ROOT="${OUT_ROOT:-/gscratch/${SLURM_JOB_ACCOUNT}/runs/cellfm/v1}"
EXP_CFG="${EXP_CFG:-configs/experiment/main_grid.yaml}"

idx=$SLURM_ARRAY_TASK_ID
if [[ $idx -ge $N_RUNS ]]; then
  echo "[hyak] array index $idx >= N_RUNS=$N_RUNS; nothing to do" >&2
  exit 0
fi
ENCODER=${ENCODERS[$idx]}
SIZE=${SIZES[$idx]}
RUN_ID=${RUN_IDS[$idx]}
BATCH_SIZE=$(batch_size_for_size "$SIZE")
TRAIN_EXTRA_ARGS=()
if [[ -n "${N_STEPS:-}" ]]; then
  TRAIN_EXTRA_ARGS+=(--n-steps "$N_STEPS")
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
  TRAIN_EXTRA_ARGS+=(--resume-from "$RESUME_FROM")
elif [[ -n "${RESUME_ROOT:-}" ]]; then
  TRAIN_EXTRA_ARGS+=(--resume-from "$RESUME_ROOT/$RUN_ID/final.pt")
fi

echo "[hyak] array task $idx: encoder=$ENCODER size=$SIZE run_id=$RUN_ID batch_size=$BATCH_SIZE"

python -m scripts.train_one \
  --cache-dir "$CACHE_DIR" \
  --encoder-config "configs/encoder/${ENCODER}.yaml" \
  --model-config   "configs/model/${SIZE}.yaml" \
  --train-config   "$EXP_CFG" \
  --run-id "$RUN_ID" \
  --out-dir "$OUT_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers 4 \
  "${TRAIN_EXTRA_ARGS[@]}" \
  --wandb-project "$WANDB_PROJECT"

echo "[hyak] done: $RUN_ID"
