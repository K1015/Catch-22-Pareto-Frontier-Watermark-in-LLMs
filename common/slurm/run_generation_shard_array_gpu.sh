#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=catch22_shard
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
RESULTS_ROOT="${RESULTS_ROOT:?RESULTS_ROOT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
BASE_START_INDEX="${BASE_START_INDEX:?BASE_START_INDEX is required}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:?TOTAL_SAMPLES is required}"
SHARD_SIZE="${SHARD_SIZE:?SHARD_SIZE is required}"
INPUT_FILE="${INPUT_FILE:-}"
MODEL_NAME="${MODEL_NAME:-}"
SEED="${SEED:-1234}"
BATCH_SAVE="${BATCH_SAVE:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
AUXILIARY_MODEL="${AUXILIARY_MODEL:-}"

shard_start=$((BASE_START_INDEX + TASK_ID * SHARD_SIZE))
if (( shard_start >= TOTAL_SAMPLES )); then
  echo "Shard start $shard_start is out of range for total samples $TOTAL_SAMPLES" >&2
  exit 1
fi

remaining=$((TOTAL_SAMPLES - shard_start))
shard_samples="$SHARD_SIZE"
if (( remaining < SHARD_SIZE )); then
  shard_samples="$remaining"
fi
shard_end=$((shard_start + shard_samples - 1))

OUTPUT_FILE="$RESULTS_ROOT/$METHOD/raw/shards/clean_${shard_start}_${shard_end}.jsonl"
SUMMARY_FILE="$RESULTS_ROOT/$METHOD/summary/shards/clean_generation_${shard_start}_${shard_end}.json"
mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$SUMMARY_FILE")"

cmd=(
  python3 "$ROOT/common/run_generation.py"
  --manifest "$MANIFEST"
  --method "$METHOD"
  --output-file "$OUTPUT_FILE"
  --summary-file "$SUMMARY_FILE"
  --start-index "$shard_start"
  --num-samples "$shard_samples"
  --seed "$SEED"
  --batch-save "$BATCH_SAVE"
)
[[ -n "$INPUT_FILE" ]] && cmd+=(--input-file "$INPUT_FILE")
[[ -n "$MODEL_NAME" ]] && cmd+=(--model-name "$MODEL_NAME")
[[ -n "$AUXILIARY_MODEL" ]] && cmd+=(--auxiliary-model "$AUXILIARY_MODEL")
[[ -n "$MAX_NEW_TOKENS" ]] && cmd+=(--max-new-tokens "$MAX_NEW_TOKENS")
[[ -n "$TEMPERATURE" ]] && cmd+=(--temperature "$TEMPERATURE")
[[ -n "$TOP_P" ]] && cmd+=(--top-p "$TOP_P")
[[ -n "$TOP_K" ]] && cmd+=(--top-k "$TOP_K")
"${cmd[@]}"
