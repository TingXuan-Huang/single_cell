# Lessons From The CellFM Hyak Run

These are observed failure modes from the CellFM setup and should be guarded
against in future Hyak repo conversions.

## Storage And Environment

- Installing Python packages into home can fail with `Disk quota exceeded`.
  Put conda envs, pip cache, raw data, processed caches, and run outputs under
  `/gscratch/<group>/<netid>/...`.
- User-site packages can leak into conda prefixes and break imports. Set
  `PYTHONNOUSERSITE=1` in Slurm jobs.
- Always print `which python` and `python --version` from inside the batch job.
- If using a conda prefix path, activate by prefix or prepend `$VENV_DIR/bin`
  after confirming `$VENV_DIR/bin/python` exists.

## Slurm Script Location

- Slurm can execute a copied script under `/var/spool/slurmd/...`. If a script
  sources helpers with `dirname "$0"`, helper lookup can fail.
- Use `PROJECT_ROOT`:

```bash
: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$PROJECT_ROOT/slurm/_common.sh"
```

## GPU Requests

- The error `Invalid GRES specification (with and without type identification)`
  can happen when a script has an untyped GPU directive and the submit command
  adds a typed GPU request.
- Use one GPU request style per job. Prefer putting GPU type in the submit
  command and leaving it out of the sbatch file.

## Memory And OOM

- 2080Ti has only about 11 GB VRAM. It is not a safe default for transformer or
  LLM-style jobs unless a probe proves the job fits.
- A40/L40/L40S are 48 GB class GPUs. H200 is the high-memory fallback.
- A model can fit after a logits-memory fix but still fail on a smaller GPU.
  Example: a 1M rank transformer failed on 2080Ti while allocating about 4.7 GB
  extra because only about 2.9 GB was free.
- Add VRAM probes or one-step smoke jobs before full arrays.
- Route arrays by measured/estimated memory, not just parameter count.

## Data Cache And Output Roots

- Reuse a completed data cache if preprocessing succeeded and code changes do
  not affect the cache schema.
- Use a new output root after training-code changes, for example `v2`, `v3`.
  This avoids mixing old `best.pt`, `final.pt`, and logs with new behavior.
- Print cache and output paths before training starts.

## Command Hygiene

- A line starting with `--export=...` is not a shell command. It must be part of
  an `sbatch` invocation.
- Give paste-ready multi-line commands with `sbatch` at the start.
- Include `mkdir -p logs "$OUT_ROOT"` before submitting jobs.

## Checkpointing And Evaluation

- Save `best.pt`, `final.pt`, and interval checkpoints if a job may be
  preempted or if embeddings over training time matter.
- Do not rely on training completion as convergence evidence. Track validation
  and downstream metrics.
- For representation learning, keep full downstream evaluation separate from
  cheap training-loop validation unless the user explicitly wants slower jobs.

## Small Result Sync

- Add an opt-in workflow to commit only small logs/results back to GitHub.
- Never stage model weights, raw data, caches, large arrays, or large logs by
  default.
- Prompt before enabling this workflow in a new repo.

## Local To Server Workflow

- Build and test a tiny local command before writing or submitting Slurm.
- Keep Slurm thin: allocate resources, activate the environment, print
  diagnostics, and call the same local shell script.
- Prefer local edits -> commit/push -> Hyak `git pull --ff-only`; if code is
  edited on Hyak, make the back-sync plan explicit.
- Use bounded checkpoint resume: `n_steps` should be a target ceiling, not an
  unbounded while loop. Resume from checkpoints only when the target step is
  larger than the checkpoint step.
- For masked-token training, make validation batches exercise the validation
  objective instead of producing all-ignored/NaN validation losses.

## Skill Mode Lessons

- The promotion ladder is only the initialization workflow. It lowers the
  chance of queue-wasting mistakes but does not solve CUDA OOM, path problems,
  quota failures, or invalid metrics after they happen.
- Preserve two creation modes: converting an existing repo into a Hyak-runnable
  repo, and designing a new repo from scratch with Hyak readiness built in.
- Treat most CellFM issues encountered here as server debugging evidence: disk
  quota, user-site package leakage, Slurm helper path, typed/untyped GPU request
  mismatch, missing cache manifest, OOM, NaN validation, and bounded resume.
- For ML projects, prefer a minimal Colab/GPU prototype when it can cheaply
  estimate batch-size and training-loop failures before Hyak queue time is spent.
