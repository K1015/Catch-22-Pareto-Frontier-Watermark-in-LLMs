#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --job-name=catch22_eval
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT="${CATCH22_ROOT:?source env.sh first}"
# shellcheck source=/dev/null
source "$ROOT/common/slurm/hpc_env.sh"

POSITIVE_FILE="${POSITIVE_FILE:?POSITIVE_FILE is required}"
NEGATIVE_FILE="${NEGATIVE_FILE:?NEGATIVE_FILE is required}"
CALIBRATION_FILE="${CALIBRATION_FILE:?CALIBRATION_FILE is required}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:?TOKENIZER_MODEL is required}"
OUTPUT_JSON="${OUTPUT_JSON:?OUTPUT_JSON is required}"
OUTPUT_CSV="${OUTPUT_CSV:-}"
CONDITION_LABEL="${CONDITION_LABEL:-}"

cmd=(
  python3 "$ROOT/common/evaluate_outputs.py"
  --positive-file "$POSITIVE_FILE"
  --negative-file "$NEGATIVE_FILE"
  --keyless-calibration-file "$CALIBRATION_FILE"
  --tokenizer-model "$TOKENIZER_MODEL"
  --output-json "$OUTPUT_JSON"
)
[[ -n "$OUTPUT_CSV" ]] && cmd+=(--output-csv "$OUTPUT_CSV")
[[ -n "$CONDITION_LABEL" ]] && cmd+=(--condition-label "$CONDITION_LABEL")
"${cmd[@]}"

