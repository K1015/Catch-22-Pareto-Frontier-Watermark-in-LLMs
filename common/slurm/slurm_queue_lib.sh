#!/bin/bash
set -euo pipefail

gpu_partition_cap() {
  echo "${GPU_PARALLEL_CAP:-${CATCH22_SLURM_MAX_GPU_JOBS:-4}}"
}

gpu_short_partition_cap() {
  echo "${GPU_SHORT_PARALLEL_CAP:-${CATCH22_SLURM_MAX_GPU_SHORT_JOBS:-2}}"
}

short_partition_cap() {
  echo "${SHORT_PARALLEL_CAP:-${CATCH22_SLURM_MAX_CPU_JOBS:-8}}"
}

active_partition_job_count() {
  local partition="$1"
  squeue -u "${USER}" -h -t RUNNING,CONFIGURING,COMPLETING -o "%P" | awk -v target="$partition" '$1 == target {count += 1} END {print count + 0}'
}

wait_for_partition_headroom() {
  local partition="$1"
  local cap="$2"
  local poll_seconds="${3:-20}"

  while true; do
    local active
    active="$(active_partition_job_count "$partition")"
    if (( active < cap )); then
      break
    fi
    echo "Waiting for $partition headroom: active=$active cap=$cap" >&2
    sleep "$poll_seconds"
  done
}

submit_with_headroom() {
  local partition="$1"
  local cap="$2"
  shift 2
  wait_for_partition_headroom "$partition" "$cap"
  "$@"
}

join_afterok() {
  local ids=()
  local item
  for item in "$@"; do
    [[ -n "$item" ]] && ids+=("$item")
  done
  if ((${#ids[@]} == 0)); then
    echo ""
    return
  fi
  local IFS=:
  echo "afterok:${ids[*]}"
}
