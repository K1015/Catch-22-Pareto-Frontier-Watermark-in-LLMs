#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --job-name=catch22_wave_ctl
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT="${CATCH22_ROOT:?source env.sh first}"
TASK_FILE="${TASK_FILE:?TASK_FILE is required}"
RUNNER="${RUNNER:?RUNNER is required}"
CHUNK_LINES="${CHUNK_LINES:-2}"
ARRAY_CAP="${ARRAY_CAP:-2}"
CHUNK_ROOT="${CHUNK_ROOT:-$(dirname "$TASK_FILE")/controller_chunks}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-catch22_wave}"
POLL_SECONDS="${POLL_SECONDS:-120}"
WAIT_PARTITION="${WAIT_PARTITION:-}"
WAIT_ACTIVE_MAX="${WAIT_ACTIVE_MAX:-}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"

mkdir -p "$ROOT/logs" "$CHUNK_ROOT"

task_count() {
  local file="$1"
  grep -cve '^[[:space:]]*$' "$file" || true
}

terminal_states_for_job() {
  local jid="$1"
  sacct -j "$jid" -n -P -o State 2>/dev/null | awk -F'|' 'NF { print $1 }' | sort -u | tr '\n' ' '
}

active_partition_jobs() {
  local partition="$1"
  squeue -h -u "${USER}" -p "$partition" -t PD,R,CF,CG -o '%i' | wc -l | tr -d '[:space:]'
}

wait_for_partition_headroom() {
  if [[ -z "$WAIT_PARTITION" || -z "$WAIT_ACTIVE_MAX" ]]; then
    return
  fi
  while true; do
    local active
    active="$(active_partition_jobs "$WAIT_PARTITION")"
    if (( active <= WAIT_ACTIVE_MAX )); then
      echo "Partition $WAIT_PARTITION has headroom: active_or_pending=$active <= $WAIT_ACTIVE_MAX"
      return
    fi
    echo "Waiting for $WAIT_PARTITION headroom: active_or_pending=$active > $WAIT_ACTIVE_MAX"
    sleep "$POLL_SECONDS"
  done
}

wait_for_job() {
  local jid="$1"
  local label="$2"
  while squeue -h -j "$jid" | grep -q .; do
    echo "Waiting for $label job $jid"
    sleep "$POLL_SECONDS"
  done

  local states
  states="$(terminal_states_for_job "$jid")"
  echo "Finished $label job $jid states: ${states:-unknown}"
  if echo "$states" | grep -Eq 'FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE'; then
    echo "$label job $jid did not complete cleanly: $states" >&2
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      exit 1
    fi
  fi
}

rm -rf "$CHUNK_ROOT"
mkdir -p "$CHUNK_ROOT"
split -l "$CHUNK_LINES" -d -a 4 "$TASK_FILE" "$CHUNK_ROOT/chunk_"
for chunk in "$CHUNK_ROOT"/chunk_*; do
  [[ -f "$chunk" ]] || continue
  mv "$chunk" "$chunk.tasks"
done

mapfile -t chunks < <(find "$CHUNK_ROOT" -maxdepth 1 -type f -name '*.tasks' | sort)
echo "Controller task file: $TASK_FILE"
echo "Runner: $RUNNER"
echo "Chunks: ${#chunks[@]} chunk_lines=$CHUNK_LINES cap=$ARRAY_CAP"

for idx in "${!chunks[@]}"; do
  chunk="${chunks[$idx]}"
  n="$(task_count "$chunk")"
  [[ "$n" -gt 0 ]] || continue
  wait_for_partition_headroom
  jid="$(sbatch --parsable --array="0-$((n - 1))%${ARRAY_CAP}" --job-name="${JOB_NAME_PREFIX}_${idx}" --export="ALL,TASK_FILE=${chunk}" "$RUNNER")"
  jid="$(printf '%s\n' "$jid" | tail -n 1 | tr -d '[:space:]')"
  echo "Submitted ${JOB_NAME_PREFIX}_${idx}: job=$jid tasks=$n file=$chunk"
  wait_for_job "$jid" "${JOB_NAME_PREFIX}_${idx}"
done

echo "Wave controller completed for $TASK_FILE"
