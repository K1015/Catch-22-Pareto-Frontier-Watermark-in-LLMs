#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --job-name=catch22_merge_score
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT="${CATCH22_ROOT:?source env.sh first}"
# shellcheck source=/dev/null
source "$ROOT/common/slurm/hpc_env.sh"

INPUT_DIR="${INPUT_DIR:?INPUT_DIR is required}"
OUTPUT_FILE="${OUTPUT_FILE:?OUTPUT_FILE is required}"
SUMMARY_FILE="${SUMMARY_FILE:?SUMMARY_FILE is required}"
EXPECTED_TOTAL="${EXPECTED_TOTAL:-}"

cmd=(
  python3 "$ROOT/common/merge_scored_shards.py"
  --input-dir "$INPUT_DIR"
  --output-file "$OUTPUT_FILE"
  --summary-file "$SUMMARY_FILE"
)
if [[ -n "$EXPECTED_TOTAL" ]]; then
  cmd+=(--expected-total "$EXPECTED_TOTAL")
fi
"${cmd[@]}"
