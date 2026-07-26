# Local To Hyak Workflow

Use this reference when converting a repo that currently works locally, or when
creating Slurm scripts before the project has been proven on Hyak. The goal is
to avoid the common HPC failure mode: works on laptop, then crashes after hours
in queue.

This is an initialization workflow. It does not by itself resolve CUDA OOM,
server path issues, quota failures, broken packages, or bad training metrics.
When a job has already failed, switch to `server-debugging.md`.

## Workflow Modes

Use one of two creation modes:

- Existing repo conversion: preserve existing repo commands and wrap them with
  portable paths, local run scripts, and thin Slurm.
- Start from scratch: scaffold the repo around this ladder from day one, so the
  same command can run locally, in Colab/minimal GPU, and on Hyak.

Use server-debugging mode separately after failures.

## Promotion Ladder

Require a bounded, progressive path before full-scale jobs:

1. Tiny local CPU test
2. Portable paths and config
3. Optional Colab/minimal GPU prototype
4. Reproducible environment
5. Local shell script that runs the real command
6. Thin Slurm wrapper around that shell script
7. Tiny Hyak interactive GPU test
8. Short batch job
9. Full-scale job or array
10. Small-result sync back to local, if the user opted in

Do not skip directly from local code to a long Slurm job unless the user
explicitly accepts the risk.

## Stage 1: Tiny Local Test

Before touching Hyak, ask for or create a command that runs on the laptop or a
CPU-only environment with tiny settings:

```bash
python train.py \
  --epochs 1 \
  --batch-size 2 \
  --subset 100 \
  --num-workers 0 \
  --output-dir runs/smoke_local
```

The local smoke test should verify imports, data loading, training/eval loop,
checkpoint save/load, metrics, and output paths. If the repo has no tiny mode,
add one before writing Slurm.

## Stage 1.5: Minimal Colab GPU Prototype

For ML/AI/science training, prefer a tiny Colab script or notebook before long
Hyak queueing when GPU memory or CUDA behavior is uncertain. Minimize
everything:

- Include only the code needed to construct the model, load or fake one tiny
  dataset shard, and run one forward/backward/update step.
- Download the smallest useful data subset, for example 100 cells, 100 images,
  10 molecules, or one mini shard.
- Print CUDA availability, GPU name, model parameter count, batch size, first
  loss, and peak memory if available.
- Try a small batch-size sweep such as 1, 2, 4, 8, stopping at first OOM.
- Do not put full preprocessing, full downloads, result analysis, or Slurm logic
  into the Colab prototype.

The Colab result is only a rough signal; Hyak still needs an interactive or
short batch smoke test because filesystem, modules, quota, and Slurm behavior
are different.

## Stage 2: Portable Paths

Reject hardcoded local paths such as `/Users/.../Desktop/data` in runnable code.
Prefer CLI args, config fields, or environment variables:

```bash
export DATA_DIR=/gscratch/<group>/<netid>/data
export OUT_ROOT=/gscratch/<group>/<netid>/runs/<project>/v1
python train.py --data-dir "$DATA_DIR" --out-dir "$OUT_ROOT"
```

In Python, use `pathlib.Path` and pass paths through config or args. Keep local
and Hyak paths different only at the environment/config layer.

## Stage 3: Reproducible Environment

Prefer one of these, depending on the repo:

```bash
conda env export --no-builds > environment.yml
python -m pip freeze > requirements.txt
```

On Hyak, install into `/gscratch/<group>/<netid>/envs/...`, not home. Set
`PYTHONNOUSERSITE=1` in Slurm to avoid user-site package leakage.

## Stage 4: Local Run Script First

Create a local script such as `scripts/run_train.sh` or `run.sh` before Slurm:

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${DATA_DIR:=data/small}"
: "${OUT_DIR:=runs/smoke}"
python train.py --config configs/base.yaml --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
```

Test it locally:

```bash
bash scripts/run_train.sh
```

Only after this works should Slurm call it.

## Stage 5: Thin Slurm Wrapper

Slurm should allocate resources, activate the environment, print diagnostics,
and call the same shell script. Keep training logic out of Slurm.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=my-job
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${PROJECT_ROOT:=${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT"
mkdir -p logs
source slurm/_common.sh
bash scripts/run_train.sh
```

The exact same command should be runnable interactively with `bash scripts/run_train.sh`.

## Stage 6: Hyak Interactive Test

Before a long job, recommend a tiny interactive session:

```bash
srun --account=<account> --partition=<gpu-partition> --gpus=1 \
  --cpus-per-task=4 --mem=16G --time=1:00:00 --pty bash
```

Inside the allocation:

```bash
cd /gscratch/<group>/<netid>/<repo>
source slurm/_common.sh
DATA_DIR=/gscratch/<group>/<netid>/data_small OUT_DIR=/gscratch/<group>/<netid>/runs/smoke_hyak bash scripts/run_train.sh
```

This catches CUDA, module, permission, quota, dataset path, and small OOM issues
without waiting for a long queued job.

## Stage 7: Short Batch Job

After interactive success, submit a short batch job using the same wrapper:

```bash
sbatch --account=<account> --partition=<gpu-partition> --gpus=1 \
  --time=02:00:00 \
  --export=ALL,DATA_DIR=/gscratch/<group>/<netid>/data_small,OUT_DIR=/gscratch/<group>/<netid>/runs/smoke_batch \
  slurm/train.sbatch
```

Check logs, checkpoint files, metrics, and `sacct` memory before scaling.

## Stage 8: Full Scale

Only scale after the short batch job proves:

- The same entry command works on Hyak.
- Environment and paths are correct.
- Logs are written under `logs/`.
- Checkpoints and resume work.
- One-step VRAM or memory probe passes with margin.
- Output root is unique for this code/config version.

For arrays, split jobs by measured resource needs rather than parameter count
alone.

## Local Server Integration

Default code flow:

```bash
# Local machine
cd /path/to/repo
git status --short
git add <changed files>
git commit -m "..."
git push origin <branch>
```

```bash
# Hyak login node
cd /gscratch/<group>/<netid>/<repo>
git pull --ff-only origin <branch>
```

If the user edits code on Hyak, warn that those changes must be committed and
pushed or copied back before local edits continue. Prefer local edits plus server
pull unless debugging requires direct server edits.

For result feedback, offer an allowlisted small-result sync script. It should
stage only small `.out`, `.err`, `.json`, `.csv`, `.tsv`, and `.md` files, never
raw data, caches, arrays, or model checkpoints.

## Required Startup Diagnostics

Every Hyak command path should print early:

```bash
pwd
hostname
git rev-parse --short HEAD || true
which python
python --version
python - <<'PYINFO'
import os
print('cwd', os.getcwd())
try:
    import torch
    print('torch', torch.__version__)
    print('cuda available', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('gpu', torch.cuda.get_device_name(0))
except Exception as e:
    print('torch check failed', repr(e))
PYINFO
```

## Beginner Mistakes To Guard Against

- Testing only on full data.
- Writing complex training logic directly in Slurm.
- No checkpointing or no resume path.
- No pinned or reproducible environment.
- Hardcoded laptop paths.
- Submitting 24-hour jobs before an interactive GPU smoke test.
- Ignoring logs and memory accounting after the first short job.
