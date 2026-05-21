# Catch-22: Pareto Frontier for Detectability and Robustness in LLM Watermarking

LLM watermarking has its own Catch-22[^catch22-name]: watermarks that are easy to verify are often easier to notice, while watermarks that stay hidden are easier to remove with edits.

[^catch22-name]: The name alludes to Joseph Heller's *Catch-22*, a paradoxical dilemma in which one decision cannot be made without negating another. In the context of LLMs, watermarks face an analogous bind: improving robustness often makes them more detectable, while reducing detectability weakens their robustness.

This repository is a standalone reproduction package for the accepted ICML 2026 paper "Catch-22: On the Fundamental Tradeoff Between Detectability and Robustness in LLM Watermarking" by Kuheli Pratihar and Debdeep Mukhopadhyay.

The experiments focus on Long-Form Question Answering (LFQA), where a model writes detailed answers to open-ended questions. The supported model setups are:

- `meta-llama/Llama-2-7b-hf`
- `mistralai/Mistral-7B-v0.1`

The included runs evaluate every implemented watermark method, including the `hybrid` method. They measure detection on clean outputs and robustness after two edit attacks: a moderate Dipper rewrite and a stronger summary-style paraphrase.

## Overview

Large language models generate text by sampling tokens, a process now widely used for inference-time watermarking that verifies AI-generated content. We present an information-theoretic framework that captures the trade-off between robustness to text edits and detectability by observers who lack the watermark key or use a keyless detector.

The bounds hold regardless of computational power, and what a keyless detector can achieve depends on what it can observe about the model and its outputs. At the heart of the analysis is an additive Kullback-Leibler (KL) information measure that quantifies how well a hypothesis test can distinguish watermarked from unwatermarked text while the watermark remains stealthy. The measure remains zero for distribution-preserving schemes and increases with text length for token-level and sentence-level probability-modifying schemes.

When edits are modeled as noise, the KL measure shrinks quadratically with the edit rate for token-level schemes and with an induced semantic flip rate for sentence-level schemes. This shrinkage exposes an unavoidable trilemma among robustness, stealth, and reliable verification. Guided by these limits, we use a hybrid watermarking strategy that selects the Pareto-optimal scheme among distribution-preserving, semantic-level, and token-level methods based on the expected editing regime at deployment.

Experiments on Llama-2-7B and Mistral-7B under paraphrasing attacks corroborate the theoretical predictions and show that the hybrid strategy lies near the Pareto frontier across the evaluated edit regimes.

![Watermarking schemes in modern LLMs exhibit a trade-off between detectability via statistical tests and robustness against LLM output editing.](figures/HLV.png)

Figure: Watermarking schemes in modern LLMs exhibit a trade-off between detectability via statistical tests and robustness against LLM output editing.

## Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{pratihar2026catch22,
  title = {Catch-22: On the Fundamental Tradeoff Between Detectability and Robustness in LLM Watermarking},
  author = {Pratihar, Kuheli and Mukhopadhyay, Debdeep},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year = {2026},
  url = {https://icml.cc/virtual/2026/poster/66807}
}
```

Paper page: https://icml.cc/virtual/2026/poster/66807

## Methods and Conditions

Watermark methods:

`kgw`, `unigram`, `dipmark`, `hcw`, `heavywater`, `simplexwater`, `kuditipudi`, `semstamp`, `pmark`, `simmark`, `cgw`, `gaussmark`, `dawa`, `hybrid`.

Default conditions:

- `clean`: no paraphrasing attack.
- `dipper_moderate`: moderate paraphrasing using the Dipper paraphraser setting.
- `extreme_paraphrase`: stronger summarization-style rewrite.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For GPU inference with quantized 7B models, install a PyTorch build matching your CUDA stack. If you use gated model checkpoints, authenticate with Hugging Face before running the full pipeline:

```bash
huggingface-cli login
```

## Data

The expected LFQA input file is:

```text
data/lfqa/inputs.jsonl
```

The repository includes a three-row example file for environment checks. Replace it with the paper LFQA prompts for full reproduction. Each row must include a `prompt` field and may include `id` and `reference`.

## Environment Check

The local backend exercises the complete pipeline without downloading Llama2, Mistral, or paraphraser models:

```bash
python -m catch22.pipeline \
  --config configs/llama2_lfqa.yaml \
  --reproduction-suite \
  --num-samples 2 \
  --local-backend \
  --resume
```

Repeat for Mistral:

```bash
python -m catch22.pipeline \
  --config configs/mistral_lfqa.yaml \
  --reproduction-suite \
  --num-samples 2 \
  --local-backend \
  --resume
```

## Full Reproduction

Run the reproduction suite for Llama2:

```bash
python -m catch22.pipeline \
  --config configs/llama2_lfqa.yaml \
  --reproduction-suite \
  --resume
```

Run the same pipeline for Mistral:

```bash
python -m catch22.pipeline \
  --config configs/mistral_lfqa.yaml \
  --reproduction-suite \
  --resume
```

Use `--model-name-or-path /path/to/local/checkpoint` when you want to use a locally downloaded checkpoint instead of the model identifier in the config.

## Individual Stages

```bash
python -m catch22.generate --config configs/llama2_lfqa.yaml --method hybrid --resume
python -m catch22.attack --config configs/llama2_lfqa.yaml --method hybrid --attack dipper --paraphrase-strength moderate --resume
python -m catch22.attack --config configs/llama2_lfqa.yaml --method hybrid --attack extreme-paraphrase --paraphrase-strength extreme --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --condition clean --resume
python -m catch22.evaluate --config configs/llama2_lfqa.yaml --method hybrid
python -m catch22.render --config configs/llama2_lfqa.yaml
```

Use `--dry-run` on any command to print resolved paths and planned work without running models.

## Outputs

Generated outputs are written under `outputs/` and are ignored by git:

```text
outputs/<track>/<method>/clean/generations.jsonl
outputs/<track>/<method>/attacks/<condition>/attacked.jsonl
outputs/<track>/<method>/scored/<condition>/scored.jsonl
outputs/<track>/<method>/evaluations/<condition>.json
outputs/<track>/tables/*.json
outputs/<track>/tables/*.tex
```
