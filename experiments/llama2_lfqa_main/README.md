# Llama-2 LFQA Main

This experiment is the authoritative reproduction path for the main Catch-22 table
on Llama-2-7B (paper Table 1, `tab:llama-main`). It runs the inference-time
watermarking pipeline (`common/` + `Llama2-Watermark/`) over the LFQA prompts in
`data/lfqa/inputs.jsonl`.

`manifest.json` is the single definition of the methods, attacks, model, and
sampling parameters for this experiment.

## What the manifest specifies

- **Model:** `meta-llama/Llama-2-7b-hf`, loaded in 4-bit, sampled with
  `max_new_tokens=300`, `temperature=0.8`, `top_p=0.95`, `top_k=50`, and
  `num_samples=500`.
- **Methods (14):** `kgw`, `unigram`, `dipmark`, `hcw`, `heavywater`,
  `simplexwater`, `kuditipudi`, `semstamp`, `pmark`, `simmark`, `cgw`,
  `gaussmark`, `dawa`, `hybrid`.
- **Attacks (7):** `none`, `dipper`, `opt`, `wm-removal`, `synonym`,
  `backtranslation`, `summarization`.
- **Rendered table:** `rendered/llama2_main_table.tex` and
  `rendered/llama2_main_table.json`, written under this experiment's results
  directory.
- **Evaluation parameters:** `vanilla` calibration, `target_fpr=0.01`, and the
  summarization attack model `facebook/bart-large-cnn`.

The method and attack names above match the registries and terminology in the
[repository README](../../README.md).

## Files in this experiment

| File | Description |
| --- | --- |
| `manifest.json` | Definition of the methods, attacks, model, sampling parameters, and table outputs. |
| `slurm/submit_all.sh` | Submits the full pipeline for every method x attack via the shared launcher. |
| `preflight_check.py` | Preflight validation script that exercises the pipeline command-line interface before a full run. |
| `methods_generation.txt` | Method list used during the generation stage. |
| `methods_preflight_chunks.txt` | Method list used during the preflight stage. |

## Running the experiment

Validate the manifest first:

```bash
source env.sh
python common/validate_experiment.py --manifest experiments/llama2_lfqa_main/manifest.json
```

Then submit the whole pipeline on a SLURM cluster — asset staging, preflight,
clean generation, attacks, scoring, keyed and keyless evaluation, and table
render:

```bash
source env.sh
experiments/llama2_lfqa_main/slurm/submit_all.sh
```

`submit_all.sh` sets `MANIFEST_REL` and `MODEL_NAME` (defaulting to
`CATCH22_MODEL_LLAMA2`) and hands off to `common/slurm/submit_lfqa_track.sh`. All
paths, partitions, account, GPU specification, and queue caps are supplied by the
`CATCH22_*` macros in `env.sh`. Launcher variables may be overridden inline, for
example `NUM_SAMPLES=50 experiments/llama2_lfqa_main/slurm/submit_all.sh`.

Rendered tables are written to `$CATCH22_RESULTS_DIR/llama2_lfqa_main/rendered/`.

For a stage-by-stage walkthrough, including single-GPU runs without SLURM, see
[`docs/reproduce_lfqa.md`](../../docs/reproduce_lfqa.md).
