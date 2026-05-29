# Reproducing the LFQA experiments

This guide assumes that [Installation](../README.md#installation) has been completed via `scripts/setup_env.sh` and that `env.sh` has been sourced. `scripts/setup_env.sh` installs `requirements.txt`, the NLTK corpora used by the attacks, and runs `scripts/fetch_external.sh` to clone the upstream method repositories into `external/`.

The full pipeline is manifest-driven. The one-command path is the per-track submit script; the stage-by-stage commands below apply to a single-GPU run without SLURM or to the isolated execution of a single method.

## 0. Validate the manifest

```bash
source env.sh
python common/validate_experiment.py --manifest experiments/mistral7b_appendix/manifest.json
python common/validate_experiment.py --manifest experiments/llama2_lfqa_main/manifest.json
```

## 1. One-command reproduction (SLURM)

```bash
source env.sh
experiments/llama2_lfqa_main/slurm/submit_all.sh      # Llama-2-7B main table
experiments/mistral7b_appendix/slurm/submit_all.sh    # Mistral-7B appendix table
```

Each submit script stages attack assets, then, for every method × attack, runs clean generation, the attack (on both watermarked and vanilla outputs), scoring, evaluation, and table rendering, chained through SLURM job dependencies. Rendered tables are written to `$CATCH22_RESULTS_DIR/<track>/rendered/`.

## 2. Stage-by-stage (single GPU, no SLURM)

The model defaults to the manifest's `model_name`; it can be overridden with `--model-name "$CATCH22_MODEL_MISTRAL"` or a local snapshot path. Each stage writes a JSONL output file and a summary JSON file.

```bash
source env.sh
M=experiments/mistral7b_appendix/manifest.json
R="$CATCH22_RESULTS_DIR/mistral7b_appendix"
METHOD=kgw

# 2a. Clean generation: the vanilla calibration baseline and the watermarked method.
python common/run_generation.py --manifest "$M" --method vanilla \
  --output-file "$R/vanilla/raw/clean.jsonl" \
  --summary-file "$R/vanilla/summary/clean_generation.json" --resume
python common/run_generation.py --manifest "$M" --method "$METHOD" \
  --output-file "$R/$METHOD/raw/clean.jsonl" \
  --summary-file "$R/$METHOD/summary/clean_generation.json" --resume

# 2b. Attack (example: DIPPER). Run once on the watermarked output (positive)
#     and once on the vanilla output (negative) to provide the AUROC baseline.
python common/run_attack_suite.py --manifest "$M" --attack-name dipper \
  --input-file "$R/$METHOD/raw/clean.jsonl" \
  --output-file "$R/$METHOD/attacks/dipper/positive_attacked.jsonl" \
  --summary-file "$R/$METHOD/attacks/dipper/positive_summary.json" \
  --attack-model kalpeshk2011/dipper-paraphraser-xxl
python common/run_attack_suite.py --manifest "$M" --attack-name dipper \
  --input-file "$R/vanilla/raw/clean.jsonl" \
  --output-file "$R/$METHOD/attacks/dipper/negative_attacked.jsonl" \
  --summary-file "$R/$METHOD/attacks/dipper/negative_summary.json" \
  --attack-model kalpeshk2011/dipper-paraphraser-xxl

# 2c. Score both attacked files with the method detector.
python common/score_outputs.py --manifest "$M" --method "$METHOD" \
  --input-file "$R/$METHOD/attacks/dipper/positive_attacked.jsonl" \
  --output-file "$R/$METHOD/scored/dipper/positive_scored.jsonl" \
  --summary-file "$R/$METHOD/scored/dipper/positive_summary.json"
python common/score_outputs.py --manifest "$M" --method "$METHOD" \
  --input-file "$R/$METHOD/attacks/dipper/negative_attacked.jsonl" \
  --output-file "$R/$METHOD/scored/dipper/negative_scored.jsonl" \
  --summary-file "$R/$METHOD/scored/dipper/negative_summary.json"

# 2d. Evaluate: AUROC, TPR@1%FPR, keyless z, edit-rate.
python common/evaluate_outputs.py \
  --positive-file "$R/$METHOD/scored/dipper/positive_scored.jsonl" \
  --negative-file "$R/$METHOD/scored/dipper/negative_scored.jsonl" \
  --keyless-calibration-file "$R/vanilla/raw/clean.jsonl" \
  --tokenizer-model "$CATCH22_MODEL_MISTRAL" \
  --output-json "$R/$METHOD/evaluations/dipper.json" \
  --condition-label dipper --target-fpr 0.01

# 2e. Render the table once all methods and conditions are evaluated.
python common/render_tables.py --manifest "$M" --results-root "$R"
```

Stages 2b–2d are repeated for each attack in the manifest (`none`, `dipper`, `opt`, `wm-removal`, `synonym`, `backtranslation`, `summarization`). For the `none` condition, the clean generations are scored directly, and stage 2b is omitted. The `backtranslation` attack takes `--forward-model`/`--backward-model` in place of `--attack-model`; `opt` and `wm-removal` take a paraphraser identifier through `--attack-model "$CATCH22_MODEL_*"`. The attack definitions are in `common/attack_registry.py`.
