#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=catch22_preflight_chunk
#SBATCH --mem=96GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT="${CATCH22_ROOT:?source env.sh first}"
# shellcheck source=/dev/null
source "$ROOT/common/slurm/hpc_env.sh"

MANIFEST="${MANIFEST:?MANIFEST is required}"
METHOD_CHUNKS_FILE="${METHOD_CHUNKS_FILE:?METHOD_CHUNKS_FILE is required}"
RESULTS_ROOT="${RESULTS_ROOT:?RESULTS_ROOT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
SEED="${SEED:-1234}"
MODEL_NAME="${MODEL_NAME:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
START_OFFSET="${START_OFFSET:-0}"

chunk_index=$((START_OFFSET + TASK_ID))
line_no=$((chunk_index + 1))
METHODS="$(sed -n "${line_no}p" "$METHOD_CHUNKS_FILE" | tr -d '[:space:]')"
if [[ -z "$METHODS" ]]; then
  echo "No methods found for task index $TASK_ID in $METHOD_CHUNKS_FILE" >&2
  exit 1
fi

OUTPUT_JSON="$RESULTS_ROOT/preflight_chunks/preflight_${chunk_index}.json"
mkdir -p "$(dirname "$OUTPUT_JSON")"

cmd=(
  python3 "$ROOT/experiments/llama2_lfqa_main/preflight_check.py"
  --manifest "$MANIFEST"
  --output-json "$OUTPUT_JSON"
  --methods "$METHODS"
  --num-samples "$NUM_SAMPLES"
  --seed "$SEED"
)
if [[ -n "$MODEL_NAME" ]]; then
  cmd+=(--model-name "$MODEL_NAME")
fi
if [[ -n "$MAX_NEW_TOKENS" ]]; then
  cmd+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi
"${cmd[@]}"
