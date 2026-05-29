#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the full Mistral-7B LFQA appendix experiment (appendix table):
# asset staging -> preflight -> generation -> attacks -> scoring ->
# evaluation -> table render, for every method x attack in the manifest.
#
# Usage:
#   source env.sh
#   experiments/mistral7b_appendix/slurm/submit_all.sh
#
# All paths, the model, partitions, account, and queue caps come from the
# CATCH22_* variables in env.sh. Override any launcher variable inline, e.g.
#   NUM_SAMPLES=50 experiments/mistral7b_appendix/slurm/submit_all.sh
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${CATCH22_ROOT:?source env.sh first}"

export MANIFEST_REL="experiments/mistral7b_appendix/manifest.json"
export MODEL_NAME="${MODEL_NAME:-${CATCH22_MODEL_MISTRAL}}"

exec "$ROOT/common/slurm/submit_lfqa_track.sh"
