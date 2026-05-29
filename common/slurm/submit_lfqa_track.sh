#!/bin/bash
set -euo pipefail

ROOT="${CATCH22_ROOT:?source env.sh first}"
GPU_PARTITION="${CATCH22_SLURM_GPU_PARTITION:-gpu}"
GPU_SHORT_PARTITION="${CATCH22_SLURM_GPU_SHORT_PARTITION:-gpu}"
CPU_PARTITION="${CATCH22_SLURM_CPU_PARTITION:-short}"
MANIFEST_REL="${MANIFEST_REL:?MANIFEST_REL is required}"
MANIFEST="$ROOT/$MANIFEST_REL"
LOG_ROOT="$ROOT/logs"
mkdir -p "$LOG_ROOT"

# shellcheck source=/dev/null
source "$ROOT/common/slurm/slurm_queue_lib.sh"

MODEL_NAME="${MODEL_NAME:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-${CATCH22_CONDA_ENV:-catch22}}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
SEED="${SEED:-1234}"
BATCH_SAVE="${BATCH_SAVE:-5}"
PREFLIGHT_NUM_SAMPLES="${PREFLIGHT_NUM_SAMPLES:-5}"
HF_CACHE="${HF_CACHE:-${CATCH22_HF_CACHE}}"
OPT_MODEL="${OPT_MODEL:-facebook/opt-2.7b}"
BART_MODEL="${BART_MODEL:-facebook/bart-large-cnn}"
DIPPER_MODEL="${DIPPER_MODEL:-kalpeshk2011/dipper-paraphraser-xxl}"
BT_FORWARD_MODEL="${BT_FORWARD_MODEL:-Helsinki-NLP/opus-mt-en-fr}"
BT_BACKWARD_MODEL="${BT_BACKWARD_MODEL:-Helsinki-NLP/opus-mt-fr-en}"

RESULTS_SUBDIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["results_subdir"])' "$MANIFEST")"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results/$RESULTS_SUBDIR}"
METHODS_CSV="${METHODS:-$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["methods"]))' "$MANIFEST")}"
ATTACKS_CSV="${ATTACKS:-$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["attacks"]))' "$MANIFEST")}"
DATASET_PATH="$(python3 -c 'import json,os,sys; data=json.load(open(sys.argv[1])); path=data.get("dataset_path"); print(os.path.normpath(os.path.join(os.path.dirname(sys.argv[1]), path)) if path else "")' "$MANIFEST")"

STAGE_SCRIPT="$ROOT/common/slurm/run_stage_assets_short.sh"
PREFLIGHT_SCRIPT="$ROOT/common/slurm/run_preflight_gpu_short.sh"
GENERATE_SCRIPT="$ROOT/common/slurm/run_generation_gpu.sh"
ATTACK_GPU_SCRIPT="$ROOT/common/slurm/run_attack_gpu.sh"
ATTACK_SHORT_SCRIPT="$ROOT/common/slurm/run_attack_short.sh"
SCORE_SCRIPT="$ROOT/common/slurm/run_score_gpu.sh"
EVAL_SCRIPT="$ROOT/common/slurm/run_eval_short.sh"
RENDER_SCRIPT="$ROOT/common/slurm/run_render_short.sh"

mkdir -p "$RESULTS_ROOT"

submit_job() {
  local partition="$1"
  local cap="$2"
  local dependency="$3"
  local export_args="$4"
  local script_path="$5"

  local cmd=(sbatch --parsable --partition="$partition")
  [[ -n "${CATCH22_SLURM_ACCOUNT:-}" ]] && cmd+=(--account="${CATCH22_SLURM_ACCOUNT}")
  [[ -n "$dependency" ]] && cmd+=(--dependency="$dependency")
  cmd+=(--export="$export_args" "$script_path")
  submit_with_headroom "$partition" "$cap" "${cmd[@]}"
}

split_csv() {
  local csv="$1"
  local old_ifs="$IFS"
  IFS=','
  read -r -a OUT <<< "$csv"
  IFS="$old_ifs"
}

stage_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},HF_CACHE=${HF_CACHE},OUTPUT_JSON=${RESULTS_ROOT}/asset_stage_summary.json"
stage_job="$(submit_job "$CPU_PARTITION" "$(short_partition_cap)" "" "$stage_export" "$STAGE_SCRIPT")"
echo "Submitted asset staging: $stage_job"

preflight_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},MANIFEST=${MANIFEST},OUTPUT_JSON=${RESULTS_ROOT}/preflight_check.json,NUM_SAMPLES=${PREFLIGHT_NUM_SAMPLES},METHODS=${METHODS_CSV}"
[[ -n "$MODEL_NAME" ]] && preflight_export+=",MODEL_NAME=${MODEL_NAME}"
[[ -n "$MAX_NEW_TOKENS" ]] && preflight_export+=",MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
preflight_job="$(submit_job "$GPU_SHORT_PARTITION" "$(gpu_short_partition_cap)" "afterok:${stage_job}" "$preflight_export" "$PREFLIGHT_SCRIPT")"
echo "Submitted preflight check: $preflight_job"

vanilla_raw="$RESULTS_ROOT/vanilla/raw/clean.jsonl"
vanilla_summary="$RESULTS_ROOT/vanilla/summary/clean_generation.json"
vanilla_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},MANIFEST=${MANIFEST},METHOD=vanilla,INPUT_FILE=${DATASET_PATH},OUTPUT_FILE=${vanilla_raw},SUMMARY_FILE=${vanilla_summary},SEED=${SEED},BATCH_SAVE=${BATCH_SAVE}"
[[ -n "$MODEL_NAME" ]] && vanilla_export+=",MODEL_NAME=${MODEL_NAME}"
[[ -n "$NUM_SAMPLES" ]] && vanilla_export+=",NUM_SAMPLES=${NUM_SAMPLES}"
[[ -n "$MAX_NEW_TOKENS" ]] && vanilla_export+=",MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
[[ -n "$TEMPERATURE" ]] && vanilla_export+=",TEMPERATURE=${TEMPERATURE}"
[[ -n "$TOP_P" ]] && vanilla_export+=",TOP_P=${TOP_P}"
[[ -n "$TOP_K" ]] && vanilla_export+=",TOP_K=${TOP_K}"
vanilla_job="$(submit_job "$GPU_PARTITION" "$(gpu_partition_cap)" "afterok:${preflight_job}" "$vanilla_export" "$GENERATE_SCRIPT")"
echo "Submitted vanilla generation: $vanilla_job"

split_csv "$METHODS_CSV"
methods=("${OUT[@]}")
split_csv "$ATTACKS_CSV"
attacks=("${OUT[@]}")

render_deps=()

for method in "${methods[@]}"; do
  method="${method// /}"
  [[ -n "$method" ]] || continue
  clean_raw="$RESULTS_ROOT/$method/raw/clean.jsonl"
  clean_summary="$RESULTS_ROOT/$method/summary/clean_generation.json"
  gen_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},MANIFEST=${MANIFEST},METHOD=${method},INPUT_FILE=${DATASET_PATH},OUTPUT_FILE=${clean_raw},SUMMARY_FILE=${clean_summary},SEED=${SEED},BATCH_SAVE=${BATCH_SAVE}"
  [[ -n "$MODEL_NAME" ]] && gen_export+=",MODEL_NAME=${MODEL_NAME}"
  [[ -n "$NUM_SAMPLES" ]] && gen_export+=",NUM_SAMPLES=${NUM_SAMPLES}"
  [[ -n "$MAX_NEW_TOKENS" ]] && gen_export+=",MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
  [[ -n "$TEMPERATURE" ]] && gen_export+=",TEMPERATURE=${TEMPERATURE}"
  [[ -n "$TOP_P" ]] && gen_export+=",TOP_P=${TOP_P}"
  [[ -n "$TOP_K" ]] && gen_export+=",TOP_K=${TOP_K}"
  clean_job="$(submit_job "$GPU_PARTITION" "$(gpu_partition_cap)" "afterok:${preflight_job}" "$gen_export" "$GENERATE_SCRIPT")"
  echo "Submitted ${method} generation: $clean_job"

  none_scored_dir="$RESULTS_ROOT/$method/scored/none"
  none_score_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},MANIFEST=${MANIFEST},METHOD=${method},INPUT_FILE=${clean_raw},OUTPUT_FILE=${none_scored_dir}/positive_scored.jsonl,SUMMARY_FILE=${none_scored_dir}/positive_summary.json,SEED=${SEED}"
  [[ -n "$MODEL_NAME" ]] && none_score_export+=",MODEL_NAME=${MODEL_NAME}"
  none_positive_score_job="$(submit_job "$GPU_PARTITION" "$(gpu_partition_cap)" "$(join_afterok "$clean_job")" "$none_score_export" "$SCORE_SCRIPT")"
  none_neg_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},MANIFEST=${MANIFEST},METHOD=${method},INPUT_FILE=${vanilla_raw},OUTPUT_FILE=${none_scored_dir}/negative_scored.jsonl,SUMMARY_FILE=${none_scored_dir}/negative_summary.json,SEED=${SEED}"
  [[ -n "$MODEL_NAME" ]] && none_neg_export+=",MODEL_NAME=${MODEL_NAME}"
  none_negative_score_job="$(submit_job "$GPU_PARTITION" "$(gpu_partition_cap)" "$(join_afterok "$vanilla_job")" "$none_neg_export" "$SCORE_SCRIPT")"
  none_eval_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},POSITIVE_FILE=${none_scored_dir}/positive_scored.jsonl,NEGATIVE_FILE=${none_scored_dir}/negative_scored.jsonl,CALIBRATION_FILE=${vanilla_raw},TOKENIZER_MODEL=${MODEL_NAME:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_name"])' "$MANIFEST")},OUTPUT_JSON=${RESULTS_ROOT}/${method}/evaluations/none.json,CONDITION_LABEL=none"
  none_eval_job="$(submit_job "$CPU_PARTITION" "$(short_partition_cap)" "$(join_afterok "$none_positive_score_job" "$none_negative_score_job")" "$none_eval_export" "$EVAL_SCRIPT")"
  render_deps+=("$none_eval_job")

  for attack in "${attacks[@]}"; do
    attack="${attack// /}"
    [[ "$attack" != "none" && -n "$attack" ]] || continue
    attack_dir="$RESULTS_ROOT/$method/attacks/$attack"
    positive_attacked="$attack_dir/positive_attacked.jsonl"
    negative_attacked="$attack_dir/negative_attacked.jsonl"
    positive_summary="$attack_dir/positive_summary.json"
    negative_summary="$attack_dir/negative_summary.json"

    attack_model=""
    forward_model=""
    backward_model=""
    attack_script="$ATTACK_GPU_SCRIPT"
    attack_partition="$GPU_PARTITION"
    attack_cap="$(gpu_partition_cap)"
    case "$attack" in
      dipper)
        attack_model="$DIPPER_MODEL"
        ;;
      opt|wm-removal)
        attack_model="$OPT_MODEL"
        ;;
      summarization)
        attack_model="$BART_MODEL"
        ;;
      backtranslation)
        attack_script="$ATTACK_SHORT_SCRIPT"
        attack_partition="$CPU_PARTITION"
        attack_cap="$(short_partition_cap)"
        forward_model="$BT_FORWARD_MODEL"
        backward_model="$BT_BACKWARD_MODEL"
        ;;
      synonym|span-synonym)
        attack_script="$ATTACK_SHORT_SCRIPT"
        attack_partition="$CPU_PARTITION"
        attack_cap="$(short_partition_cap)"
        ;;
    esac

    positive_attack_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},MANIFEST=${MANIFEST},ATTACK_NAME=${attack},INPUT_FILE=${clean_raw},OUTPUT_FILE=${positive_attacked},SUMMARY_FILE=${positive_summary}"
    negative_attack_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},MANIFEST=${MANIFEST},ATTACK_NAME=${attack},INPUT_FILE=${vanilla_raw},OUTPUT_FILE=${negative_attacked},SUMMARY_FILE=${negative_summary}"
    if [[ -n "$MODEL_NAME" ]]; then
      positive_attack_export+=",SOURCE_MODEL=${MODEL_NAME}"
      negative_attack_export+=",SOURCE_MODEL=${MODEL_NAME}"
    fi
    [[ -n "$NUM_SAMPLES" ]] && positive_attack_export+=",NUM_SAMPLES=${NUM_SAMPLES}" && negative_attack_export+=",NUM_SAMPLES=${NUM_SAMPLES}"
    [[ -n "$attack_model" ]] && positive_attack_export+=",ATTACK_MODEL=${attack_model}" && negative_attack_export+=",ATTACK_MODEL=${attack_model}"
    [[ -n "$forward_model" ]] && positive_attack_export+=",FORWARD_MODEL=${forward_model}" && negative_attack_export+=",FORWARD_MODEL=${forward_model}"
    [[ -n "$backward_model" ]] && positive_attack_export+=",BACKWARD_MODEL=${backward_model}" && negative_attack_export+=",BACKWARD_MODEL=${backward_model}"

    positive_attack_job="$(submit_job "$attack_partition" "$attack_cap" "$(join_afterok "$clean_job")" "$positive_attack_export" "$attack_script")"
    negative_attack_job="$(submit_job "$attack_partition" "$attack_cap" "$(join_afterok "$vanilla_job")" "$negative_attack_export" "$attack_script")"

    scored_dir="$RESULTS_ROOT/$method/scored/$attack"
    positive_score_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},MANIFEST=${MANIFEST},METHOD=${method},INPUT_FILE=${positive_attacked},OUTPUT_FILE=${scored_dir}/positive_scored.jsonl,SUMMARY_FILE=${scored_dir}/positive_summary.json,SEED=${SEED}"
    negative_score_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},MANIFEST=${MANIFEST},METHOD=${method},INPUT_FILE=${negative_attacked},OUTPUT_FILE=${scored_dir}/negative_scored.jsonl,SUMMARY_FILE=${scored_dir}/negative_summary.json,SEED=${SEED}"
    [[ -n "$MODEL_NAME" ]] && positive_score_export+=",MODEL_NAME=${MODEL_NAME}" && negative_score_export+=",MODEL_NAME=${MODEL_NAME}"
    positive_score_job="$(submit_job "$GPU_PARTITION" "$(gpu_partition_cap)" "$(join_afterok "$positive_attack_job")" "$positive_score_export" "$SCORE_SCRIPT")"
    negative_score_job="$(submit_job "$GPU_PARTITION" "$(gpu_partition_cap)" "$(join_afterok "$negative_attack_job")" "$negative_score_export" "$SCORE_SCRIPT")"

    eval_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},POSITIVE_FILE=${scored_dir}/positive_scored.jsonl,NEGATIVE_FILE=${scored_dir}/negative_scored.jsonl,CALIBRATION_FILE=${vanilla_raw},TOKENIZER_MODEL=${MODEL_NAME:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_name"])' "$MANIFEST")},OUTPUT_JSON=${RESULTS_ROOT}/${method}/evaluations/${attack}.json,CONDITION_LABEL=${attack}"
    eval_job="$(submit_job "$CPU_PARTITION" "$(short_partition_cap)" "$(join_afterok "$positive_score_job" "$negative_score_job")" "$eval_export" "$EVAL_SCRIPT")"
    render_deps+=("$eval_job")
  done
done

render_export="ALL,ROOT=${ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},MANIFEST=${MANIFEST},RESULTS_ROOT=${RESULTS_ROOT}"
render_job="$(submit_job "$CPU_PARTITION" "$(short_partition_cap)" "$(join_afterok "${render_deps[@]}")" "$render_export" "$RENDER_SCRIPT")"
echo "Submitted final render: $render_job"
echo "All LFQA jobs submitted for $MANIFEST_REL"
