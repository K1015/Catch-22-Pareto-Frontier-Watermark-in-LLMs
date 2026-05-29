#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --job-name=catch22_score_short
#SBATCH --mem=64GB
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
INPUT_FILE="${INPUT_FILE:?INPUT_FILE is required}"
OUTPUT_FILE="${OUTPUT_FILE:?OUTPUT_FILE is required}"
SUMMARY_FILE="${SUMMARY_FILE:?SUMMARY_FILE is required}"
MODEL_NAME="${MODEL_NAME:-}"
AUXILIARY_MODEL="${AUXILIARY_MODEL:-}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
BATCH_SAVE="${BATCH_SAVE:-20}"
SEED="${SEED:-1234}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
RESUME="${RESUME:-0}"

cmd=(
  python3 "$ROOT/common/score_outputs.py"
  --manifest "$MANIFEST"
  --method "$METHOD"
  --input-file "$INPUT_FILE"
  --output-file "$OUTPUT_FILE"
  --summary-file "$SUMMARY_FILE"
  --seed "$SEED"
  --batch-save "$BATCH_SAVE"
  --torch-dtype "$TORCH_DTYPE"
)
[[ -n "$MODEL_NAME" ]] && cmd+=(--model-name "$MODEL_NAME")
[[ -n "$AUXILIARY_MODEL" ]] && cmd+=(--auxiliary-model "$AUXILIARY_MODEL")
[[ -n "$NUM_SAMPLES" ]] && cmd+=(--num-samples "$NUM_SAMPLES")
[[ "$RESUME" == "1" ]] && cmd+=(--resume)
"${cmd[@]}"
