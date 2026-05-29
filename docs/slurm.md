# SLURM launchers

The pipeline ships working SLURM launchers under `common/slurm/` and per-track
entry points under `experiments/<track>/slurm/submit_all.sh`. Every path,
partition, account, and queue cap is read from the `CATCH22_*` macros in your
`env.sh` — no script contains a user name, cluster name, or absolute path.

## How configuration flows

1. You `source env.sh` in your login shell. This exports `CATCH22_ROOT`,
   `CATCH22_HF_CACHE`, the SLURM site macros, etc.
2. `submit_all.sh` sets the manifest and model, then calls
   `common/slurm/submit_lfqa_track.sh`, the orchestrator.
3. The orchestrator submits every stage with `sbatch --export=ALL,...`, so the
   `CATCH22_*` variables propagate into each job.
4. Each `run_*.sh` sources `common/slurm/hpc_env.sh`, which re-sources
   `env.sh`, activates the conda env (if configured), and exports the HF cache
   and thread settings.

## Site macros (set in `env.sh`)

| Macro | Used for |
| --- | --- |
| `CATCH22_SLURM_GPU_PARTITION` | GPU generation / attack / scoring jobs (`sbatch --partition`) |
| `CATCH22_SLURM_GPU_SHORT_PARTITION` | Short GPU preflight jobs |
| `CATCH22_SLURM_CPU_PARTITION` | CPU-only staging / eval / render / lexical attacks |
| `CATCH22_SLURM_ACCOUNT` | `sbatch --account` (omitted if empty) |
| `CATCH22_SLURM_GRES` / `CATCH22_SLURM_MEM` / `CATCH22_SLURM_CPUS` | GPU spec, memory, cores (see the `#SBATCH` defaults in each `run_*.sh`) |
| `CATCH22_SLURM_MAX_GPU_JOBS` / `..._GPU_SHORT_JOBS` / `..._CPU_JOBS` | Queue-headroom caps enforced by `slurm_queue_lib.sh` |

The orchestrator injects `--partition` (and `--account`) onto every `sbatch`
call from these macros, overriding the generic defaults baked into the
`#SBATCH` directives. To change GPU spec, wall-time, or memory, either edit the
`#SBATCH` lines in the relevant `run_*.sh` template or pass an `sbatch`
override when submitting.

## Running a full track

```bash
source env.sh
experiments/llama2_lfqa_main/slurm/submit_all.sh
experiments/mistral7b_appendix/slurm/submit_all.sh
```

Common inline overrides (any launcher variable can be exported first):

```bash
# smaller pilot run
NUM_SAMPLES=50 experiments/mistral7b_appendix/slurm/submit_all.sh

# use a locally staged model snapshot instead of the HF id
MODEL_NAME="$CATCH22_HF_CACHE/models--mistralai--Mistral-7B-v0.1/snapshots/<hash>" \
  experiments/mistral7b_appendix/slurm/submit_all.sh
```

## Individual stage jobs

The `run_*.sh` scripts can also be submitted directly with the environment
variables each one documents at the top, e.g.:

```bash
source env.sh
sbatch --export=ALL,MANIFEST="$CATCH22_ROOT/experiments/mistral7b_appendix/manifest.json",\
METHOD=kgw,INPUT_FILE=...,OUTPUT_FILE=...,SUMMARY_FILE=... \
  common/slurm/run_generation_gpu.sh
```

Logs are written to `logs/<job-name>-<job-id>.out|err` under `CATCH22_ROOT`
(created automatically by `hpc_env.sh`).

## Queue policy

`slurm_queue_lib.sh` throttles submissions so the number of concurrently
active jobs per partition never exceeds your `CATCH22_SLURM_MAX_*` caps. Tune
these to match your site's QOS limits; the defaults (4 GPU / 2 GPU-short / 8
CPU) are conservative.
