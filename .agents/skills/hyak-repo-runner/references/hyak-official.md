# Official Hyak Reference

Use these UW Hyak docs as primary sources when designing Slurm files and
submit workflows.

## Sources

- Start Here: https://hyak.uw.edu/docs/compute/start-here/
- Scheduling Jobs: https://hyak.uw.edu/docs/compute/scheduling-jobs/
- Using Idle Resources / Checkpoint: https://hyak.uw.edu/docs/compute/checkpoint/
- GPU Start: https://hyak.uw.edu/docs/gpus/gpu_start/
- Resource Monitoring: https://hyak.uw.edu/docs/compute/resource-monitoring/

## Facts To Preserve

- Hyak uses Slurm. Users submit work through Slurm jobs instead of running heavy
  work on login nodes.
- `hyakalloc` shows available account/partition combinations, contributed
  resources, utilization, and checkpoint availability.
- `sinfo -s` summarizes partitions. `sinfo -p ckpt-all -O nodehost,cpusstate,freemem,gres,gresused -S nodehost | grep -v null`
  helps inspect checkpoint GPUs.
- Account and partition are separate Slurm concepts. Use `--account=<account>`
  and `--partition=<partition>`.
- Typical batch script fields include account, partition, nodes, tasks/cores,
  memory, GPU request, time, chdir, output, and error.
- Hyak docs show `--gpus=<count>` for owned GPU partitions and
  `--gpus-per-node=<type>:<count>` for checkpoint typed GPU requests.
- Checkpoint partitions (`ckpt`, `ckpt-g2`, `ckpt-all`) use idle resources.
  Idle now does not guarantee immediate availability.
- Checkpoint jobs may be preempted/requeued. Jobs should save progress and be
  able to resume from checkpoints.
- Hyak docs recommend setting `--time` to the expected maximum runtime with
  margin.
- GPU memory classes listed by Hyak:
  - 2080 Ti: 11 GB
  - Titan: 24 GB
  - A100: 40 GB
  - A40, L40, L40S, RTX6k: 48 GB
  - H200: 141 GB
- Common Slurm error from Hyak docs: `Exceeded job memory limit` means the job
  used more memory than requested.

## Commands Worth Emitting

```bash
hyakalloc
sinfo -s
sinfo -p ckpt-all -O nodehost,cpusstate,freemem,gres,gresused -S nodehost | grep -v null
squeue -u "$USER"
sacct -j <jobid> --format=JobID,JobName%30,State,Elapsed,ReqMem,MaxRSS,AllocTRES%50
```
