#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --job-name=catch22_stage_assets
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT="${CATCH22_ROOT:?source env.sh first}"
# shellcheck source=/dev/null
source "$ROOT/common/slurm/hpc_env.sh"

HF_CACHE="${HF_CACHE:-$HF_HOME}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/results/asset_stage_summary.json}"

python3 "$ROOT/common/stage_attack_assets.py" \
  --hf-cache "$HF_CACHE" \
  --output-json "$OUTPUT_JSON"

