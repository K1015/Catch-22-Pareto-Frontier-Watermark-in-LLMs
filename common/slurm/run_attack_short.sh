#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --job-name=catch22_attack_short
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT="${CATCH22_ROOT:?source env.sh first}"
# shellcheck source=/dev/null
source "$ROOT/common/slurm/hpc_env.sh"

MANIFEST="${MANIFEST:?MANIFEST is required}"
ATTACK_NAME="${ATTACK_NAME:?ATTACK_NAME is required}"
INPUT_FILE="${INPUT_FILE:?INPUT_FILE is required}"
OUTPUT_FILE="${OUTPUT_FILE:?OUTPUT_FILE is required}"
SUMMARY_FILE="${SUMMARY_FILE:?SUMMARY_FILE is required}"
SOURCE_MODEL="${SOURCE_MODEL:-}"
ATTACK_MODEL="${ATTACK_MODEL:-}"
FORWARD_MODEL="${FORWARD_MODEL:-}"
BACKWARD_MODEL="${BACKWARD_MODEL:-}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
BATCH_SAVE="${BATCH_SAVE:-5}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
TARGET_EDIT_RATE="${TARGET_EDIT_RATE:-0.15}"
DIPPER_LEXICAL_DIVERSITY="${DIPPER_LEXICAL_DIVERSITY:-40}"
DIPPER_ORDER_DIVERSITY="${DIPPER_ORDER_DIVERSITY:-0}"
DIPPER_SENT_INTERVAL="${DIPPER_SENT_INTERVAL:-3}"
SPAN_MIN_LENGTH="${SPAN_MIN_LENGTH:-5}"
SPAN_MAX_LENGTH="${SPAN_MAX_LENGTH:-10}"

cmd=(
  python3 "$ROOT/common/run_attack_suite.py"
  --manifest "$MANIFEST"
  --attack-name "$ATTACK_NAME"
  --input-file "$INPUT_FILE"
  --output-file "$OUTPUT_FILE"
  --summary-file "$SUMMARY_FILE"
  --torch-dtype "$TORCH_DTYPE"
  --batch-save "$BATCH_SAVE"
  --attack-batch-size "$ATTACK_BATCH_SIZE"
  --target-edit-rate "$TARGET_EDIT_RATE"
  --dipper-lexical-diversity "$DIPPER_LEXICAL_DIVERSITY"
  --dipper-order-diversity "$DIPPER_ORDER_DIVERSITY"
  --dipper-sent-interval "$DIPPER_SENT_INTERVAL"
  --span-min-length "$SPAN_MIN_LENGTH"
  --span-max-length "$SPAN_MAX_LENGTH"
)
[[ -n "$NUM_SAMPLES" ]] && cmd+=(--num-samples "$NUM_SAMPLES")
[[ -n "$SOURCE_MODEL" ]] && cmd+=(--source-model "$SOURCE_MODEL")
[[ -n "$ATTACK_MODEL" ]] && cmd+=(--attack-model "$ATTACK_MODEL")
[[ -n "$FORWARD_MODEL" ]] && cmd+=(--forward-model "$FORWARD_MODEL")
[[ -n "$BACKWARD_MODEL" ]] && cmd+=(--backward-model "$BACKWARD_MODEL")
"${cmd[@]}"
