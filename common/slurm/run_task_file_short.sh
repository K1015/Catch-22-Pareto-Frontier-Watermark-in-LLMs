#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --job-name=catch22_task_short
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

TASK_FILE="${TASK_FILE:?TASK_FILE is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
line_no=$((TASK_ID + 1))
task_cmd="$(sed -n "${line_no}p" "$TASK_FILE")"

if [[ -z "$task_cmd" ]]; then
  echo "No task found for task index $TASK_ID in $TASK_FILE" >&2
  exit 1
fi

echo "Running task $TASK_ID from $TASK_FILE"
echo "$task_cmd"
bash -lc "$task_cmd"
