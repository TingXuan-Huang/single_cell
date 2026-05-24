#!/usr/bin/env bash
# End-to-end orchestrator for the v1 sweep on Hyak.
# Submits jobs with proper dependencies:
#   1) download
#   2) build_cache         (after download)
#   3) pca_baseline        (after cache)        — CPU
#   4) scvi_baseline       (after cache)        — GPU
#   5) train_array  (16 runs, after cache)
#   6) eval_array   (after train_array + pca_baseline)
#   7) compare      (after eval_array + scvi_baseline)
#
# Adjust the SBATCH flags inside each .sbatch (account, partition) before use.

set -euo pipefail

mkdir -p logs

J_DL=$(sbatch --parsable slurm/download.sbatch)
echo "[hyak] download    -> $J_DL"

J_CACHE=$(sbatch --parsable --dependency=afterok:"$J_DL" slurm/build_cache.sbatch)
echo "[hyak] build_cache -> $J_CACHE (after $J_DL)"

J_PCA=$(sbatch --parsable --dependency=afterok:"$J_CACHE" slurm/pca_baseline.sbatch)
echo "[hyak] pca64       -> $J_PCA (after $J_CACHE)"

J_SCVI=$(sbatch --parsable --dependency=afterok:"$J_CACHE" slurm/scvi_baseline.sbatch)
echo "[hyak] scvi32      -> $J_SCVI (after $J_CACHE)"

J_TRAIN=$(sbatch --parsable --dependency=afterok:"$J_CACHE" slurm/train_array.sbatch)
echo "[hyak] train_array -> $J_TRAIN (after $J_CACHE)"

J_EVAL=$(sbatch --parsable --dependency=afterok:"$J_TRAIN":"$J_PCA" slurm/eval_array.sbatch)
echo "[hyak] eval_array  -> $J_EVAL (after $J_TRAIN, $J_PCA)"

J_CMP=$(sbatch --parsable --dependency=afterok:"$J_EVAL":"$J_SCVI" slurm/compare.sbatch)
echo "[hyak] compare     -> $J_CMP (after $J_EVAL, $J_SCVI)"
