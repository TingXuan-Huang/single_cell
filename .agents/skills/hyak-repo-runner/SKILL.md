---
name: hyak-repo-runner
description: >-
  Use when turning a code repository into a UW Hyak runnable project or
  debugging a failed Hyak/Klone server run: Slurm sbatch files, submit
  wrappers, environment activation, GPU/CPU resource planning, checkpointing,
  monitoring, local-to-server git sync, Colab/minimal GPU prototypes, and small
  result sync. Trigger for Hyak, Klone, Slurm, sbatch, hyakalloc, checkpoints,
  CUDA out of memory, server path issues, failed job logs, or GPU partition
  planning.
---

# Hyak Repo Runner

Turn a code repo into a Hyak-runnable project with reproducible Slurm scripts,
resource-aware submit commands, and a small-artifact feedback loop.

Use this skill for UW Hyak/Klone jobs, especially ML training, batch sweeps,
GPU partition selection, checkpoint jobs, and "make this repo runnable on Hyak"
requests.

## Load References

Before editing files or proposing commands, read:

- `references/hyak-official.md` for official UW Hyak facts and source links.
- `references/local-to-hyak-workflow.md` when converting a repo, adding Slurm,
  designing local-to-server run commands, or starting a workflow from scratch.
- `references/server-debugging.md` when a Hyak job has already failed, OOMed,
  found a path/env issue, or produced invalid metrics.
- `references/chat-lessons.md` for pitfalls encountered in the CellFM run.

Load additional project docs only after inspecting the repo.

## Operating Rule

Do not assume the repo is ready for Slurm. First discover the entrypoints,
dependencies, data paths, expected outputs, checkpoint behavior, and resource
shape. Then design the smallest stable Hyak workflow.

Treat the promotion ladder as an initialization workflow, not as a complete
solution to server failures. CUDA OOM, path errors, quota failures, package
leakage, Slurm request errors, missing caches, and NaN metrics require server
debugging from logs and minimal reproducers.

Always ask or explicitly confirm this design choice before scaffolding:

> Should I add a small-artifact sync workflow that commits only logs, `.json`,
> `.csv`, `.tsv`, `.md`, and other allowlisted small files back to GitHub so the
> local repo can pull results from Hyak?

Default recommendation: yes, but keep it allowlisted and size-capped. Never
auto-commit model checkpoints, raw data, caches, `.h5ad`, `.pt`, `.ckpt`,
`.npy`, `.npz`, or large logs unless the user explicitly asks.

Prefer this development direction unless the user says otherwise: edit locally,
commit and push, then `git pull --ff-only` on Hyak. If code is edited on Hyak,
make the back-sync plan explicit before more local edits continue.

## Modes

Choose the mode before editing files or giving commands:

- Existing repo conversion: inspect current entrypoints and add local scripts,
  Slurm wrappers, environment setup, and submit commands around the repo's
  existing patterns. This is the original purpose of the skill.
- Greenfield Hyak-ready scaffold: when starting from scratch, design the repo so
  the same command runs locally, in Colab/minimal GPU, and on Hyak.
- Server debugging: when the user provides errors, logs, failed job ids, CUDA
  OOM, path issues, or bad metrics, read `references/server-debugging.md` and
  diagnose the failure instead of rebuilding the workflow generically.
- Colab mini-prototype: for ML/AI jobs with uncertain GPU memory or training
  behavior, design a minimal Colab script/notebook that downloads the smallest
  possible data, runs only the training path, and estimates batch-size/training
  errors before Hyak scaling.

## Workflow

### 0. Start With The Promotion Ladder

For new Hyak conversions, design the workflow in this order:

1. Tiny local CPU command.
2. Portable paths through args/config/env vars.
3. Optional Colab/minimal GPU prototype for ML training risk.
4. Reproducible environment.
5. Local shell script that runs the actual workload.
6. Thin Slurm wrapper that only allocates resources and calls the shell script.
7. Tiny Hyak interactive GPU run.
8. Short batch job.
9. Full-scale job or array.
10. Optional small-result sync back to GitHub.

Read `references/local-to-hyak-workflow.md` for the detailed pattern and command
templates. Do not create a long-running Slurm job before there is a tiny command
that proves imports, paths, data loading, checkpointing, and metrics. Remember
that this ladder reduces risk but does not replace server debugging for OOM,
path, environment, or numeric failures.

### 1. Inspect The Repo

Identify:

- Language/runtime: Python, R, Julia, compiled binary, container, mixed.
- Main commands: train, eval, preprocess, test, smoke run, tiny local run.
- Dependency path: conda/env, venv, modules, container, lock files.
- Data/cache needs and storage size.
- Output paths and checkpoint names.
- Whether jobs are single task, array tasks, dependency chain, or multi-node.
- Current git remote and branch, plus local-to-Hyak pull/push workflow.

Prefer existing repo patterns. If scripts already exist, patch them rather than
replacing them.

### 2. Ask For Hyak Basics

If not already known, ask for:

- Hyak account, for example `stf`.
- Repo path on Hyak, for example `/gscratch/<group>/<netid>/<repo>`.
- Storage group and netid.
- Desired partitions or GPU types.
- Whether checkpoint preemption is acceptable.
- Whether logs/small results should sync to GitHub.

If the user gives partial data, make conservative assumptions and state them.

### 3. Inventory Resources

Tell the user to run, or include in the runbook:

```bash
hyakalloc
sinfo -s
sinfo -p ckpt-all -O nodehost,cpusstate,freemem,gres,gresused -S nodehost | grep -v null
```

Use official Hyak GPU memory classes as first-pass planning numbers:

- 2080 Ti: 11 GB
- A100: 40 GB
- A40, L40, L40S, RTX6k: 48 GB
- H200: 141 GB

Apply a safety margin. Do not place unknown transformer/LLM jobs on 2080Ti
unless a probe proves they fit.

### 4. Estimate Before Running

Add one of these before full training:

- `--dry-run`, `--max-steps 1`, or `--smoke` path if the repo supports it.
- A Colab/minimal GPU prototype for ML jobs when Hyak queue time is expensive:
  include only the code needed to run one tiny training step, download or mock
  the smallest useful data subset, and report CUDA availability, peak memory,
  batch size, and first loss values.
- A VRAM probe that runs one forward/backward batch and prints peak memory.
- Batch-size sweep from small to target, stopping on OOM.
- CPU memory check for preprocessing/caching.

For ML jobs, require:

- Logs print model size, batch size, sequence length, device, CUDA visibility,
  data/cache path, output root, and git commit.
- Checkpoints save often enough for checkpoint preemption.
- New output root per major code change, such as `runs/<project>/v2`.

### 5. Scaffold Local Scripts And Slurm Files

Create or patch a local shell script first, for example `scripts/run_train.sh`,
that can run outside Slurm with tiny settings. Slurm should call this script
instead of duplicating the training command.

### 6. Scaffold Slurm Files

Create or update a `slurm/` directory. Typical files:

```text
slurm/
  _common.sh
  _runs.sh                 # optional array mapping
  train_array.sbatch       # array job
  eval_array.sbatch        # optional dependent eval
  preprocess.sbatch        # optional data/cache build
  submit_hyak.sh           # one-command submission wrapper
  sync_small_results.sh    # optional, ask first
```

Rules:

- Source helpers through `PROJECT_ROOT`, not `dirname "$0"`, because Slurm may
  execute a copied script from `/var/spool/slurmd`.
- Keep paths configurable through exported environment variables.
- Use `set -euo pipefail`.
- Print all resolved paths and the active Python before running the workload.
- Create `logs/` and output directories before `sbatch`.
- Do not hardcode a user home path for data, cache, or large outputs.
- Prefer `/gscratch/<group>/<netid>/...` for environments, caches, runs, and
  pip caches to avoid home quota failures.

Example `_common.sh` behavior:

```bash
: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/gscratch/${SLURM_JOB_ACCOUNT:-stf}/${USER}/pip-cache}"

if [[ -n "${VENV_DIR:-}" && -x "$VENV_DIR/bin/python" ]]; then
  export PATH="$VENV_DIR/bin:$PATH"
fi

echo "[hyak] PROJECT_ROOT=$PROJECT_ROOT"
echo "[hyak] python=$(command -v python) ($(python --version 2>&1))"
echo "[hyak] host=$(hostname) cuda=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo none)"
```

### 7. Avoid GPU Request Mistakes

Use one GPU request style per job. Do not mix an untyped `#SBATCH --gres=gpu:1`
inside the sbatch file with a typed CLI request like `--gpus-per-node=a40:1`.

Preferred pattern:

- Keep GPU type out of the sbatch file.
- Put GPU type in the submit command or submit wrapper.

Examples:

```bash
# Owned GPU partition, generic one GPU.
sbatch --account=<acct> --partition=gpu-l40s --gpus=1 slurm/train_array.sbatch

# Checkpoint typed GPU.
sbatch --account=<acct> --partition=ckpt-all --gpus-per-node=h200:1 slurm/train_array.sbatch
```

If the local site accepts `--gres=gpu:<type>:1`, it is fine, but keep it
consistent and do not combine typed and untyped GPU directives.

### 8. Submit With A Time-Saving Wrapper

For repeated workflows, create `slurm/submit_hyak.sh` that:

- Sets `PROJECT_ROOT`, `VENV_DIR`, `CACHE_DIR`, `OUT_ROOT`, and project env vars.
- Creates logs and output dirs.
- Runs a smoke/probe job first when needed.
- Splits arrays by resource class.
- Prints exact `squeue`, `tail`, and `sacct` follow-up commands.

For checkpoint jobs, design for requeue/preemption: save checkpoints and make
resume behavior explicit.

### 9. Monitor And Triage

Include commands:

```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,JobName%30,State,Elapsed,ReqMem,MaxRSS,AllocTRES%50
tail -f logs/<pattern>.err
```

Common triage: if any of these appear, switch to server-debugging mode and
read `references/server-debugging.md`.


- `Disk quota exceeded`: move env/cache/pip cache from home to `/gscratch`.
- `No module named ...` despite conda: set `PYTHONNOUSERSITE=1`, reinstall in
  the active prefix, and print `which python`.
- `No such file ... _common.sh`: source via `$PROJECT_ROOT/slurm/_common.sh`.
- `Invalid GRES specification`: remove mixed typed/untyped GPU directives.
- CUDA OOM: reduce batch size, sequence length, workers/pin memory, or route to
  larger GPU. Do not just retry on the same GPU.
- `QOSGrpMemLimit`: lower memory request or choose a partition/account with
  free memory.

### 10. Sync Small Logs And Results

Only add this after confirming with the user. The script should:

- Pull latest before committing.
- Stage only allowlisted small files from configured dirs.
- Enforce a size cap, usually 1-5 MB per file.
- Never stage checkpoints, caches, raw data, arrays, parquet, h5ad, or model
  weights by default.
- Commit to a clearly named branch or the current branch only after user
  approval.

Safe default command shape:

```bash
git pull --ff-only origin "$(git branch --show-current)"
find logs results runs -type f \
  \( -name '*.out' -o -name '*.err' -o -name '*.json' -o -name '*.csv' -o -name '*.tsv' -o -name '*.md' \) \
  -size -2M -print0 2>/dev/null | xargs -0 git add
git status --short
git commit -m "chore: sync hyak logs and small results"
git push origin HEAD
```

Tell the user to pull locally after the server push:

```bash
git pull --ff-only origin "$(git branch --show-current)"
```

## Deliverables

When completing a Hyak conversion, provide:

- Files changed or added.
- Exact local smoke command.
- Exact local-to-Hyak push/pull commands.
- Exact setup commands for Hyak.
- Exact submit commands.
- Resource split rationale.
- Monitoring commands.
- Expected output paths.
- Failure recovery commands.
- Whether small-result sync was added or intentionally skipped.

Do not end with vague "run sbatch". Give paste-ready commands.
