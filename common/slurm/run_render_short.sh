#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --job-name=catch22_render
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT="${CATCH22_ROOT:?source env.sh first}"
# shellcheck source=/dev/null
source "$ROOT/common/slurm/hpc_env.sh"

MANIFEST="${MANIFEST:?MANIFEST is required}"
RESULTS_ROOT="${RESULTS_ROOT:-}"
OUTPUT_TEX="${OUTPUT_TEX:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

cmd=(python3 "$ROOT/common/render_tables.py" --manifest "$MANIFEST")
[[ -n "$RESULTS_ROOT" ]] && cmd+=(--results-root "$RESULTS_ROOT")
[[ -n "$OUTPUT_TEX" ]] && cmd+=(--output-tex "$OUTPUT_TEX")
[[ -n "$OUTPUT_JSON" ]] && cmd+=(--output-json "$OUTPUT_JSON")
"${cmd[@]}"

