#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=catch22_task_gshort
#SBATCH --mem=96GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-16}"
export CAUSAL_LM_FORCE_CUDA="${CAUSAL_LM_FORCE_CUDA:-1}"
export SEQ2SEQ_FORCE_CUDA="${SEQ2SEQ_FORCE_CUDA:-1}"

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
