# Mistral-7B Appendix Track

This track reproduces the Mistral-7B appendix table from the Catch-22 paper. It is kept separate from the Llama-2-7B main track (`experiments/llama2_lfqa_main/`) so that the appendix results can be regenerated independently of the Llama-2 runs.

The reproduction pipeline resides in `common/` and `Llama2-Watermark/` (the per-method inference wrappers are model-agnostic and shared by both models), driven through the manifest in this experiment directory. Dependencies are installed by `scripts/setup_env.sh`, which installs `requirements.txt`, the required NLTK corpora, and the upstream method repositories via `scripts/fetch_external.sh`.

## Configuration

The complete definition of this experiment — model, methods, attacks, sampling parameters, and table specification — is contained in [`manifest.json`](manifest.json).

- **Model:** `mistralai/Mistral-7B-v0.1` (resolved at run time from `CATCH22_MODEL_MISTRAL`)
- **Dataset:** `../../data/lfqa/inputs.jsonl`, `num_samples = 500`
- **Sampling:** `max_new_tokens = 300`, `temperature = 0.8`, `top_p = 0.95`, `top_k = 50`, `load_in_4bit = true`
- **Evaluation:** `target_fpr = 0.01`, `calibration_method = vanilla`, summarization attack model `facebook/bart-large-cnn`
- **Rendered output:** `rendered/mistral_appendix_table.tex` and `rendered/mistral_appendix_table.json` (table label `tab:mistral-eval`)

## Methods

The manifest evaluates twelve watermarking methods:

`kgw`, `unigram`, `dipmark`, `hcw`, `heavywater`, `simplexwater`, `semstamp`, `pmark`, `simmark`, `cgw`, `gaussmark`, `hybrid`.

The [methods table](../../README.md#methods-and-conditions) in the root README lists the corresponding families and references. The `kuditipudi` and `dawa` methods are not included in this track's manifest and are not run here.

`methods_generation.txt` lists the same twelve methods together with the `vanilla` unwatermarked baseline used for clean generation and calibration.

## Attacks

The manifest applies seven attack conditions, defined in `common/attack_registry.py`:

`none`, `dipper`, `opt`, `wm-removal`, `synonym`, `backtranslation`, `summarization`.

## Running

Validate the manifest, then submit the full pipeline (asset staging, clean generation, attacks, scoring, keyed and keyless evaluation, and table rendering) for every method and attack combination:

```bash
source env.sh
python common/validate_experiment.py --manifest experiments/mistral7b_appendix/manifest.json
experiments/mistral7b_appendix/slurm/submit_all.sh
```

`slurm/submit_all.sh` is the entrypoint for this track. It sets `MANIFEST_REL` and `MODEL_NAME` (from `CATCH22_MODEL_MISTRAL`) and delegates to the shared launcher `common/slurm/submit_lfqa_track.sh`. All paths, partitions, account, GPU specification, and queue caps are supplied by the `CATCH22_*` variables in `env.sh`. Individual launcher variables may be overridden inline, for example:

```bash
NUM_SAMPLES=50 experiments/mistral7b_appendix/slurm/submit_all.sh
```

Rendered tables are written under `$CATCH22_RESULTS_DIR/mistral7b_appendix/rendered/`. Stage-by-stage commands for single-GPU runs without SLURM are documented in [`docs/reproduce_lfqa.md`](../../docs/reproduce_lfqa.md), and the launcher and macro reference is documented in [`docs/slurm.md`](../../docs/slurm.md).

## Files

| Path | What it is |
| --- | --- |
| `manifest.json` | Definition of the experiment: model, methods, attacks, sampling, and table specification. |
| `methods_generation.txt` | Method list, including the `vanilla` baseline, for the generation wave. |
| `slurm/submit_all.sh` | Submits the full appendix pipeline through the shared LFQA launcher. |
