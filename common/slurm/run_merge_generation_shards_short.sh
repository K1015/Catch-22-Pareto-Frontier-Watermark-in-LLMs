#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --job-name=catch22_merge
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT="${CATCH22_ROOT:?source env.sh first}"
# shellcheck source=/dev/null
source "$ROOT/common/slurm/hpc_env.sh"

SHARDS_DIR="${SHARDS_DIR:?SHARDS_DIR is required}"
OUTPUT_FILE="${OUTPUT_FILE:?OUTPUT_FILE is required}"
SUMMARY_FILE="${SUMMARY_FILE:?SUMMARY_FILE is required}"
SHARD_GLOB="${SHARD_GLOB:-clean_*.jsonl}"
EXPECTED_TOTAL="${EXPECTED_TOTAL:-}"

mapfile -d '' shard_files < <(find "$SHARDS_DIR" -maxdepth 1 -type f -name "$SHARD_GLOB" -print0 | sort -z -V)
if ((${#shard_files[@]} == 0)); then
  echo "No shard files found in $SHARDS_DIR matching $SHARD_GLOB" >&2
  exit 1
fi

cmd=(
  python3 "$ROOT/common/merge_generation_shards.py"
  --input-files "${shard_files[@]}"
  --output-file "$OUTPUT_FILE"
  --summary-file "$SUMMARY_FILE"
)
if [[ -n "$EXPECTED_TOTAL" ]]; then
  cmd+=(--expected-total "$EXPECTED_TOTAL")
fi
"${cmd[@]}"
