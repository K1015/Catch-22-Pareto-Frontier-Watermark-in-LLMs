#!/bin/bash
# ---------------------------------------------------------------------------
# Shared environment for all SLURM launchers.
#
# Every path here comes from the CATCH22_* variables defined in the repo-root
# `env.sh` (copied from `env.example.sh`). Nothing is hard-coded to a specific
# user, cluster, or filesystem. Source `env.sh` in your shell before you
# submit jobs; SLURM propagates the variables to the job via `--export=ALL`.
# ---------------------------------------------------------------------------
set -euo pipefail

# Self-locate the repository root if CATCH22_ROOT was not already exported.
_hpc_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
: "${CATCH22_ROOT:=$(cd "$_hpc_env_dir/../.." && pwd)}"

# Load user configuration (env.sh preferred; fall back to the tracked example).
if [[ -f "$CATCH22_ROOT/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$CATCH22_ROOT/env.sh"
elif [[ -f "$CATCH22_ROOT/env.example.sh" ]]; then
  # shellcheck source=/dev/null
  source "$CATCH22_ROOT/env.example.sh"
fi

ROOT="$CATCH22_ROOT"
cd "$ROOT"

# Optional conda activation (skipped cleanly if not configured / not present).
if [[ -n "${CATCH22_CONDA_SH:-}" && -f "${CATCH22_CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CATCH22_CONDA_SH}"
  conda activate "${CATCH22_CONDA_ENV:-catch22}" || true
fi

export HF_HOME="${CATCH22_HF_CACHE:-$ROOT/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_TF="${USE_TF:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_FLOAT32_MATMUL_PRECISION="${TORCH_FLOAT32_MATMUL_PRECISION:-high}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "${CATCH22_LOG_DIR:-$ROOT/logs}"
