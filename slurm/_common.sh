#!/usr/bin/env bash
# Common environment setup for UW Hyak (klone) SLURM jobs.
# Source this from every job script. Picks up either a project venv
# (preferred, see notes/HYAK_RUNBOOK.md) or a conda env named $CONDA_ENV.
set -euo pipefail

# --- User config (override in your job script or env) ----------------------
: "${PROJECT_ROOT:=$HOME/LLM_from_scratch}"
: "${VENV_DIR:=$PROJECT_ROOT/.venv}"
: "${CONDA_ENV:=cellfm}"
: "${WANDB_PROJECT:=cellfm-v1}"
: "${HF_HOME:=$PROJECT_ROOT/.hf_cache}"
: "${PYTHONUNBUFFERED:=1}"

# --- Module + python env ---------------------------------------------------
module purge || true
# Newer drivers prefer cuda/12.4; fall back to 12.1 if not present.
module load cuda/12.4 2>/dev/null || module load cuda/12.1 || true
module load python/3.11.4 2>/dev/null || true

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  echo "[hyak] activated venv: $VENV_DIR"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  echo "[hyak] activated conda env: $CONDA_ENV"
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  echo "[hyak] activated conda env: $CONDA_ENV"
else
  echo "[hyak] WARNING: no venv at $VENV_DIR and no conda found; using system python." >&2
fi

export PROJECT_ROOT WANDB_PROJECT HF_HOME PYTHONUNBUFFERED
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "[hyak] PROJECT_ROOT=$PROJECT_ROOT"
echo "[hyak] python=$(which python)  ($(python --version 2>&1))"
echo "[hyak] HOST=$(hostname)  CUDA=$(command -v nvidia-smi >/dev/null && nvidia-smi -L || echo none)"
