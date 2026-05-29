#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=catch22_preflight
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
OUTPUT_JSON="${OUTPUT_JSON:?OUTPUT_JSON is required}"
METHODS="${METHODS:-}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
MODEL_NAME="${MODEL_NAME:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"

cmd=(
  python3 "$ROOT/experiments/llama2_lfqa_main/preflight_check.py"
  --manifest "$MANIFEST"
  --output-json "$OUTPUT_JSON"
  --num-samples "$NUM_SAMPLES"
)
if [[ -n "$MODEL_NAME" ]]; then
  cmd+=(--model-name "$MODEL_NAME")
fi
if [[ -n "$MAX_NEW_TOKENS" ]]; then
  cmd+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi
if [[ -n "$METHODS" ]]; then
  cmd+=(--methods "$METHODS")
fi
"${cmd[@]}"
