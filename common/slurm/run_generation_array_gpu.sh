#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=catch22_gen_wave
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
METHODS_FILE="${METHODS_FILE:?METHODS_FILE is required}"
RESULTS_ROOT="${RESULTS_ROOT:?RESULTS_ROOT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
START_OFFSET="${START_OFFSET:-0}"
INPUT_FILE="${INPUT_FILE:-}"
MODEL_NAME="${MODEL_NAME:-}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
SEED="${SEED:-1234}"
BATCH_SAVE="${BATCH_SAVE:-5}"
AUXILIARY_MODEL="${AUXILIARY_MODEL:-}"
RESUME="${RESUME:-0}"

line_no=$((START_OFFSET + TASK_ID + 1))
METHOD="$(sed -n "${line_no}p" "$METHODS_FILE" | tr -d '[:space:]')"
if [[ -z "$METHOD" ]]; then
  echo "No method found for task index $TASK_ID in $METHODS_FILE" >&2
  exit 1
fi

OUTPUT_FILE="$RESULTS_ROOT/$METHOD/raw/clean.jsonl"
SUMMARY_FILE="$RESULTS_ROOT/$METHOD/summary/clean_generation.json"

cmd=(
  python3 "$ROOT/common/run_generation.py"
  --manifest "$MANIFEST"
  --method "$METHOD"
  --output-file "$OUTPUT_FILE"
  --summary-file "$SUMMARY_FILE"
  --seed "$SEED"
  --batch-save "$BATCH_SAVE"
)
[[ -n "$INPUT_FILE" ]] && cmd+=(--input-file "$INPUT_FILE")
[[ -n "$MODEL_NAME" ]] && cmd+=(--model-name "$MODEL_NAME")
[[ -n "$AUXILIARY_MODEL" ]] && cmd+=(--auxiliary-model "$AUXILIARY_MODEL")
[[ -n "$NUM_SAMPLES" ]] && cmd+=(--num-samples "$NUM_SAMPLES")
[[ -n "$MAX_NEW_TOKENS" ]] && cmd+=(--max-new-tokens "$MAX_NEW_TOKENS")
[[ -n "$TEMPERATURE" ]] && cmd+=(--temperature "$TEMPERATURE")
[[ -n "$TOP_P" ]] && cmd+=(--top-p "$TOP_P")
[[ -n "$TOP_K" ]] && cmd+=(--top-k "$TOP_K")
[[ "$RESUME" == "1" ]] && cmd+=(--resume)
"${cmd[@]}"
