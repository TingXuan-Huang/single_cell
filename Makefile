# Makefile — local convenience targets.
#
# All real compute happens on UW Hyak via slurm/*.sbatch. The targets here
# are for CPU-only smoke tests and lint/format on your laptop. Do *not*
# add a training target here — that's slurm/train_array.sbatch.

PY ?= python
PYTEST ?= pytest

.PHONY: help smoke smoke-fast lint format clean

help:
	@echo "Targets:"
	@echo "  smoke       Run the full pytest suite on synthetic data (CPU only)."
	@echo "  smoke-fast  Run the trainer smoke test only (~5 s, no model build)."
	@echo "  lint        Ruff check (no fixes)."
	@echo "  format      Ruff format + import sort."
	@echo "  clean       Remove __pycache__ and pytest cache."
	@echo ""
	@echo "For training on Hyak see notes/HYAK_RUNBOOK.md."

smoke:
	$(PYTEST) tests/ -q

smoke-fast:
	$(PYTEST) tests/test_trainer_smoke.py tests/test_synthetic.py -q

lint:
	ruff check src tests scripts

format:
	ruff check --fix src tests scripts
	ruff format src tests scripts

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
