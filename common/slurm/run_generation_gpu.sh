#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=catch22_generate
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
METHOD="${METHOD:?METHOD is required}"
OUTPUT_FILE="${OUTPUT_FILE:?OUTPUT_FILE is required}"
SUMMARY_FILE="${SUMMARY_FILE:?SUMMARY_FILE is required}"
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
START_INDEX="${START_INDEX:-}"
RESUME="${RESUME:-0}"

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
[[ -n "$START_INDEX" ]] && cmd+=(--start-index "$START_INDEX")
[[ -n "$NUM_SAMPLES" ]] && cmd+=(--num-samples "$NUM_SAMPLES")
[[ -n "$MAX_NEW_TOKENS" ]] && cmd+=(--max-new-tokens "$MAX_NEW_TOKENS")
[[ -n "$TEMPERATURE" ]] && cmd+=(--temperature "$TEMPERATURE")
[[ -n "$TOP_P" ]] && cmd+=(--top-p "$TOP_P")
[[ -n "$TOP_K" ]] && cmd+=(--top-k "$TOP_K")
[[ "$RESUME" == "1" ]] && cmd+=(--resume)
"${cmd[@]}"
