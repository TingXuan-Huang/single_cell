#!/usr/bin/env bash
# Shared run-list for slurm/train_array.sbatch and slurm/eval_array.sbatch.
# Single source of truth -> no risk of index drift between train and eval arrays.
# Keep RUN_IDS / ENCODERS / SIZES in lock-step with configs/experiment/main_grid.yaml.

# 4 encoders x 4 sizes = 16 runs.
RUN_IDS=(
  hvg_dense_tiny_1m
  embedding_bag_tiny_1m
  rank_tiny_1m
  value_bin_tiny_1m
  hvg_dense_tiny_3m
  embedding_bag_tiny_3m
  rank_tiny_3m
  value_bin_tiny_3m
  hvg_dense_tiny_5m
  embedding_bag_tiny_5m
  rank_tiny_5m
  value_bin_tiny_5m
  hvg_dense_tiny_10m
  embedding_bag_tiny_10m
  rank_tiny_10m
  value_bin_tiny_10m
)

ENCODERS=(
  hvg_dense    embedding_bag    rank    value_bin
  hvg_dense    embedding_bag    rank    value_bin
  hvg_dense    embedding_bag    rank    value_bin
  hvg_dense    embedding_bag    rank    value_bin
)

SIZES=(
  tiny_1m   tiny_1m   tiny_1m   tiny_1m
  tiny_3m   tiny_3m   tiny_3m   tiny_3m
  tiny_5m   tiny_5m   tiny_5m   tiny_5m
  tiny_10m  tiny_10m  tiny_10m  tiny_10m
)

N_RUNS=${#RUN_IDS[@]}

# Sanity: all three arrays must be the same length (catches F1 drift early).
if [[ ${#ENCODERS[@]} -ne $N_RUNS || ${#SIZES[@]} -ne $N_RUNS ]]; then
  echo "[slurm/_runs.sh] ERROR: RUN_IDS/ENCODERS/SIZES length mismatch" >&2
  echo "  RUN_IDS=$N_RUNS ENCODERS=${#ENCODERS[@]} SIZES=${#SIZES[@]}" >&2
  exit 1
fi

# Per-size batch-size policy. Mirrors configs/experiment/main_grid.yaml
# `train_overrides_by_size`. tiny_10m drops to 64 to fit a single A40 at L=2048.
batch_size_for_size() {
  case "$1" in
    tiny_10m) echo 64 ;;
    *)        echo 128 ;;
  esac
}
