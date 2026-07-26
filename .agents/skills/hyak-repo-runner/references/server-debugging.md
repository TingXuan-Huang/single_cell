# Hyak Server Debugging

Use this reference after a job fails, hangs, produces bad metrics, or behaves
differently on Hyak than locally. Do not treat the promotion ladder as a fix for
runtime failures; switch into debugging mode and diagnose from evidence.

## Debugging Workflow

1. Collect exact evidence before changing code:
   - `logs/*.err`, `logs/*.out`, Slurm job id, `sacct` output, `squeue` state.
   - Submit command and exported env vars.
   - Git commit on local and Hyak.
   - Active Python path, package versions, CUDA/GPU line, `pwd`, cache/data path.
2. Classify the failure: environment, path/storage, Slurm request, CPU RAM,
   GPU VRAM, data/cache, training numerics, checkpoint/resume, or sync drift.
3. Apply the smallest targeted fix.
4. Re-run the smallest reproducer: local tiny, Colab mini GPU, Hyak interactive,
   or short batch depending on where the failure occurs.
5. Only then scale the job again.

## Common Failure Classes

### CUDA Out Of Memory

Symptoms include `torch.OutOfMemoryError`, allocator messages, or probe peaks
near total VRAM. Do not blindly retry on the same GPU.

Actions:

- Record GPU type and visible memory from logs.
- Reduce batch size, sequence length, model size, precision, or expensive heads.
- Add a one-batch forward/backward VRAM probe and route arrays by measured VRAM.
- Try a larger GPU only after estimating whether the memory gap is reasonable.
- For transformer MLM heads, check whether the code materializes full
  `(batch, sequence, vocab)` logits instead of masked-only logits.

### CPU RAM Or Slurm Memory Limit

Symptoms include `oom_kill`, `Exceeded job memory limit`, or `QOSGrpMemLimit`.

Actions:

- Use `sacct -j <jobid> --format=JobID,State,ReqMem,MaxRSS,Elapsed`.
- Lower requested memory if account QOS is blocked, or choose a partition with
  free memory.
- Stream/preprocess in chunks; avoid concatenating huge AnnData/DataFrames when
  a chunked cache is possible.

### Path Or Slurm Spool Issues

Symptoms include missing `_common.sh`, missing data/cache paths, or paths that
still point to laptop directories.

Actions:

- Source helper scripts through `$PROJECT_ROOT/slurm/...`, not `dirname "$0"`.
- Print and verify `PROJECT_ROOT`, `CACHE_DIR`, `OUT_ROOT`, and `pwd` in logs.
- Replace hardcoded `/Users/...` paths with CLI args, config, or env vars.

### Environment And Package Leakage

Symptoms include `ModuleNotFoundError`, imports resolving from `~/.local`, or
broken package dependency versions.

Actions:

- Print `which python`, `python --version`, and key package versions.
- Set `PYTHONNOUSERSITE=1`.
- Install into `/gscratch/<group>/<netid>/envs/...`, not home.
- Move pip/conda caches off home if quota is tight.

### Slurm GPU Request Errors

Symptoms include `Invalid GRES specification (with and without type identification)`.

Actions:

- Use one GPU request style only.
- Avoid mixing untyped `#SBATCH --gres=gpu:1` in the file with typed CLI GPU
  requests such as `--gpus-per-node=h200:1`.
- Prefer keeping GPU type in the submit command or wrapper.

### Data Or Cache Missing

Symptoms include missing `manifest.json`, missing shards, or empty post-filter
datasets.

Actions:

- Verify the cache root and expected marker files before training.
- Print dataset sizes after each preprocessing/filtering step.
- If metadata joins drop all rows, log candidate join keys and match counts.

### Training Numerics And Bad Validation

Symptoms include finite train loss but `nan` validation, loss spikes, or early
stopping not firing.

Actions:

- Check whether validation exercises the same objective. For masked-token
  objectives, all-ignored validation targets can produce meaningless NaNs.
- Log component losses separately.
- Add finite-value assertions for validation metrics used by early stopping.
- Reduce LR or disable AMP only after confirming the NaN source.

### Checkpoint And Resume

Symptoms include jobs ending at `n_steps`, requeued jobs restarting from scratch,
or continuation commands not loading state.

Actions:

- Save model, optimizer, scheduler, scaler, step, and config state.
- Make `n_steps` a bounded target ceiling. Resume from `step+1` through
  `n_steps`; never use an unbounded loop.
- Refuse resume when checkpoint step is already greater than or equal to target
  steps, and tell the user to increase the target.

## Minimal Reproducer Choices

- Local CPU tiny run: best for imports, paths, config parsing, and checkpoint
  mechanics.
- Colab mini GPU run: best for quick CUDA/training-loop/batch-size exploration
  without waiting in queue.
- Hyak interactive GPU: best for module, filesystem, quota, account, and CUDA
  differences specific to Hyak.
- Short batch: best final check for Slurm logging, output paths, `sacct`, and
  checkpoint files.
