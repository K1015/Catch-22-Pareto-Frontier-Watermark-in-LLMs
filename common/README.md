# `common/` — shared reproduction pipeline

This package holds the method-agnostic pipeline that produces the paper's
Llama-2-7B main table and Mistral-7B appendix table. Together with
[`Llama2-Watermark/`](../Llama2-Watermark) (the per-method inference wrappers)
and [`experiments/`](../experiments) (the manifests and submit scripts), it
constitutes the reproduction path for the reported results.

## How these files are used

Every module here is executed as a script on the repository's `PYTHONPATH`,
not as an installed console entry point. Sourcing `env.sh` (and the SLURM
launchers' `slurm/hpc_env.sh`) places `CATCH22_ROOT` on `PYTHONPATH`, so the
`common.*` imports resolve and each script is invoked directly:

```bash
source env.sh
python common/run_generation.py     --manifest experiments/llama2_lfqa_main/manifest.json --method kgw
python common/validate_experiment.py --manifest experiments/llama2_lfqa_main/manifest.json
```

All paths are supplied through `CATCH22_*` macros (see
[`env.example.sh`](../env.example.sh)); no module hard-codes a machine,
account, or absolute path.

## Registries and specs

| File | Role |
| --- | --- |
| `method_registry.py` | `MethodSpec` table (`METHOD_SPECS`) mapping each preset (`kgw`, `unigram`, `dipmark`, `hcw`, `heavywater`/`simplexwater`, `kuditipudi`, `semstamp`, `pmark`, `simmark`, `cgw`, `gaussmark`, `dawa`, `hybrid`, and others) to its family, display name, engine, inference script, and runtime keyword arguments. |
| `attack_registry.py` | `AttackSpec` table (`ATTACK_SPECS`) for `none`, `dipper`, `opt`, `wm-removal`, `synonym`, `span-synonym`, `backtranslation`, and `summarization`, recording the nominal edit rate, SLURM partition, and any attack or translation model each condition requires. |
| `manifests.py` | Loads an experiment `manifest.json` into an `ExperimentManifest` (methods × attacks, model, sampling parameters, table specification) and resolves dataset paths. |

## Watermark backends

| File | Role |
| --- | --- |
| `watermark_adapters.py` | `WatermarkExperimentAdapter`, which bridges the registry to the concrete `Llama2-Watermark/` inference scripts (`SCRIPT_PATHS`), driving generation and native verification for each method. |
| `dawa_watermark.py` | Reference `Watermark` implementation for the distribution-adaptive (`dawa`) scheme. |
| `dawa_utils.py` | Model and tokenizer loading and torch-dtype helpers shared by the DAWA backend and the attacks. |

## Pipeline stages

| File | Role |
| --- | --- |
| `stage_attack_assets.py` | Pre-stages attack models and NLTK corpora into `CATCH22_HF_CACHE` before the compute jobs run. |
| `run_generation.py` | Clean (un-attacked) generation for one method and track; writes consolidated JSONL and a summary. |
| `attacks.py` | Attack implementations: identity, DIPPER, causal-LM paraphrase (OPT and watermark-removal), synonym and span-synonym substitution, back-translation, and summarization. |
| `run_attack_suite.py` | Applies one attack condition from `attacks.py` to a method's clean generations. |
| `score_outputs.py` | Re-scores clean or attacked outputs with each method's native verifier, emitting per-sample detector scores. |
| `evaluate_outputs.py` | Aggregates scores into the reported metrics: keyed **AUROC**, **TPR@1%FPR** (`--target-fpr 0.01`), the keyless **z-score** detector, and realized **edit-rate**. |
| `render_tables.py` | Renders the paper-facing LaTeX and JSON tables from the consolidated summaries. |
| `build_hybrid_selector_results.py` | Materializes the paper's Hybrid result tree by selecting per-family representatives so the standard renderer can consume it; the Hybrid method is a family selector, not a logit-level blend. |

## Merge and validation helpers

| File | Role |
| --- | --- |
| `merge_generation_shards.py` | Merges sharded generation JSONL into one raw file and a summary. |
| `merge_scored_shards.py` | Merges sharded scored outputs (`shard_*.jsonl`) into one scored file and a summary. |
| `validate_experiment.py` | Preflight and postflight checks on a manifest and its materialized artifacts (row counts, expected files). |
| `io_utils.py` | Shared I/O and metric primitives: JSONL reading, tokenization, `compute_auroc_tpr`, z-score and unigram-surprisal helpers, and CSV writers. |

## SLURM launchers — `slurm/`

The `slurm/` set is the cluster launcher layer used by the per-experiment
`submit_all.sh` scripts. Partition, account, GRES, memory, and queue caps are
all supplied by `CATCH22_SLURM_*` macros.

| File | Role |
| --- | --- |
| `hpc_env.sh` | Shared environment sourced by every launcher: self-locates `CATCH22_ROOT`, sources `env.sh`, activates the configured conda environment, and exports the HF cache and `PYTHONPATH`. |
| `slurm_queue_lib.sh` | Queue-headroom and concurrency-throttling helpers that respect the `CATCH22_SLURM_MAX_*_JOBS` caps. |
| `submit_lfqa_track.sh` | Orchestrates a full track: asset staging, preflight, generation, attacks, scoring, evaluation, and table render, wired with SLURM job dependencies. |
| `run_stage_assets_short.sh` | Runs `stage_attack_assets.py`. |
| `run_preflight_gpu_short.sh`, `run_preflight_methods_array_gpu_short.sh` | Per-method preflight checks (single job and array). |
| `run_generation_gpu.sh`, `run_generation_array_gpu.sh`, `run_generation_shard_array_gpu.sh` | Clean generation (single, array, and sharded array). |
| `run_merge_generation_shards_short.sh` | Merges generation shards. |
| `run_attack_gpu.sh`, `run_attack_short.sh` | Run an attack condition on the GPU or CPU partition. |
| `run_score_gpu.sh`, `run_score_short.sh` | Native re-scoring. |
| `run_merge_scored_shards_short.sh` | Merges scored shards. |
| `run_eval_short.sh` | Runs `evaluate_outputs.py`. |
| `run_render_short.sh` | Runs `render_tables.py`. |
| `run_task_file_gpu.sh`, `run_task_file_gpu_short.sh`, `run_task_file_short.sh`, `run_task_wave_controller_short.sh` | Generic task-file runners and the wave controller that batches queued work under the concurrency caps. |

See [`docs/reproduce_lfqa.md`](../docs/reproduce_lfqa.md) for stage-by-stage
commands (including SLURM-free single-GPU runs) and
[`docs/slurm.md`](../docs/slurm.md) for the launcher and macro reference.
