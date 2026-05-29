#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the full Llama-2-7B LFQA main experiment (Table 1 of the paper):
# asset staging -> preflight -> generation -> attacks -> scoring ->
# evaluation -> table render, for every method x attack in the manifest.
#
# Usage:
#   source env.sh
#   experiments/llama2_lfqa_main/slurm/submit_all.sh
#
# All paths, the model, partitions, account, and queue caps come from the
# CATCH22_* variables in env.sh. Override any launcher variable inline, e.g.
#   NUM_SAMPLES=50 experiments/llama2_lfqa_main/slurm/submit_all.sh
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${CATCH22_ROOT:?source env.sh first}"

export MANIFEST_REL="experiments/llama2_lfqa_main/manifest.json"
export MODEL_NAME="${MODEL_NAME:-${CATCH22_MODEL_LLAMA2}}"

exec "$ROOT/common/slurm/submit_lfqa_track.sh"
