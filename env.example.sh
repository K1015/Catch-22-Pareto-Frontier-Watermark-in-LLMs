#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Catch-22 reproduction environment.
#
# Copy this file to `env.sh` and edit the values for YOUR machine or cluster,
# then `source env.sh` before running anything:
#
#     cp env.example.sh env.sh
#     # edit env.sh
#     source env.sh
#
# Every path and site-specific setting used by the Python pipeline and the
# SLURM launchers is derived from the CATCH22_* variables below. Nothing in
# this repository hard-codes a user name, cluster name, or absolute path:
# the experiments run entirely from the values you set here.
#
# `env.sh` is git-ignored so your local paths never get committed.
# ---------------------------------------------------------------------------
set -a  # export everything defined below

# --- Repository root -------------------------------------------------------
# Auto-detected as the directory containing this file. Override only if you
# source this from an unusual location.
if [ -z "${CATCH22_ROOT:-}" ]; then
  CATCH22_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi

# --- Working directories (defaults are repo-relative; change freely) -------
CATCH22_DATA_DIR="${CATCH22_DATA_DIR:-$CATCH22_ROOT/data}"
CATCH22_OUTPUT_DIR="${CATCH22_OUTPUT_DIR:-$CATCH22_ROOT/outputs}"
CATCH22_RESULTS_DIR="${CATCH22_RESULTS_DIR:-$CATCH22_ROOT/results}"
CATCH22_LOG_DIR="${CATCH22_LOG_DIR:-$CATCH22_ROOT/logs}"
CATCH22_EXTERNAL_DIR="${CATCH22_EXTERNAL_DIR:-$CATCH22_ROOT/external}"

# Hugging Face / model cache. Point this at a large shared scratch area on a
# cluster; the repo-relative default is fine for a single workstation.
CATCH22_HF_CACHE="${CATCH22_HF_CACHE:-$CATCH22_ROOT/hf_cache}"

# --- Python environment ----------------------------------------------------
# If you use conda, set CATCH22_CONDA_SH to your conda profile script and
# CATCH22_CONDA_ENV to the environment name. If you use a plain venv, leave
# CATCH22_CONDA_SH empty and make sure the venv is already active.
CATCH22_CONDA_SH="${CATCH22_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CATCH22_CONDA_ENV="${CATCH22_CONDA_ENV:-catch22}"
CATCH22_PYTHON="${CATCH22_PYTHON:-python3}"

# --- Models (Hugging Face IDs or local snapshot paths) ---------------------
# Set these to local snapshot directories under $CATCH22_HF_CACHE if your
# compute nodes have no internet access.
CATCH22_MODEL_LLAMA2="${CATCH22_MODEL_LLAMA2:-meta-llama/Llama-2-7b-hf}"
CATCH22_MODEL_MISTRAL="${CATCH22_MODEL_MISTRAL:-mistralai/Mistral-7B-v0.1}"

# --- SLURM site configuration (only needed for cluster runs) ---------------
# Partition names, account, GPU spec and time limits are 100% site-specific.
# Fill in the values your scheduler expects; leave a value empty to omit the
# corresponding sbatch flag.
CATCH22_SLURM_GPU_PARTITION="${CATCH22_SLURM_GPU_PARTITION:-gpu}"
CATCH22_SLURM_GPU_SHORT_PARTITION="${CATCH22_SLURM_GPU_SHORT_PARTITION:-gpu}"
CATCH22_SLURM_CPU_PARTITION="${CATCH22_SLURM_CPU_PARTITION:-short}"
CATCH22_SLURM_ACCOUNT="${CATCH22_SLURM_ACCOUNT:-}"           # e.g. --account=<name>; empty = omit
CATCH22_SLURM_GRES="${CATCH22_SLURM_GRES:-gpu:1}"            # e.g. gpu:1 or gpu:h100:1
CATCH22_SLURM_GPU_TIME="${CATCH22_SLURM_GPU_TIME:-08:00:00}"
CATCH22_SLURM_GPU_SHORT_TIME="${CATCH22_SLURM_GPU_SHORT_TIME:-02:00:00}"
CATCH22_SLURM_CPU_TIME="${CATCH22_SLURM_CPU_TIME:-08:00:00}"
CATCH22_SLURM_MEM="${CATCH22_SLURM_MEM:-96GB}"
CATCH22_SLURM_CPUS="${CATCH22_SLURM_CPUS:-4}"
# Max concurrently running array tasks per partition (your queue policy).
CATCH22_SLURM_MAX_GPU_JOBS="${CATCH22_SLURM_MAX_GPU_JOBS:-4}"
CATCH22_SLURM_MAX_GPU_SHORT_JOBS="${CATCH22_SLURM_MAX_GPU_SHORT_JOBS:-2}"
CATCH22_SLURM_MAX_CPU_JOBS="${CATCH22_SLURM_MAX_CPU_JOBS:-8}"

# --- Optional: remote sync target for `scripts/sync_to_cluster.sh` ---------
# Format: user@host:/absolute/path/on/cluster/  -- leave empty if unused.
CATCH22_REMOTE="${CATCH22_REMOTE:-}"

set +a

# Make sure the standard working directories exist.
mkdir -p "$CATCH22_DATA_DIR" "$CATCH22_OUTPUT_DIR" "$CATCH22_RESULTS_DIR" \
         "$CATCH22_LOG_DIR" "$CATCH22_HF_CACHE" "$CATCH22_EXTERNAL_DIR" 2>/dev/null || true
