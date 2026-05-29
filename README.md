# Catch-22: Pareto Frontier for Detectability and Robustness in LLM Watermarking

LLM watermarking has its own Catch-22[^catch22-name]: watermarks that are easy to verify are often easier to notice, while watermarks that stay hidden are easier to remove with edits.

[^catch22-name]: The name alludes to Joseph Heller's *Catch-22*, a paradoxical dilemma in which one decision cannot be made without negating another. In the context of LLMs, watermarks face an analogous bind: improving robustness often makes them more detectable, while reducing detectability weakens their robustness.

This repository is a standalone reproduction package for the accepted ICML 2026 paper "Catch-22: On the Fundamental Tradeoff Between Detectability and Robustness in LLM Watermarking" by Kuheli Pratihar and Debdeep Mukhopadhyay.

The experiments focus on Long-Form Question Answering (LFQA), where a model writes detailed answers to open-ended questions. The supported model setups are:

- `meta-llama/Llama-2-7b-hf`
- `mistralai/Mistral-7B-v0.1`

The included runs evaluate every implemented watermark method, including the `hybrid` method. They measure detection on clean outputs and robustness after two edit attacks: a moderate Dipper rewrite and a stronger summary-style paraphrase. Detectability summaries include detector score, z-score, detection rate, and AUROC against unwatermarked `vanilla` outputs.

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
@inproceedings{catch22watermarking2026,
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

Watermark references:

- `kgw`: Kirchenbauer et al., ["A Watermark for Large Language Models"](https://openreview.net/pdf?id=aX8ig9X2a7), ICML 2023.
- `unigram`: Zhao et al., ["Provable Robust Watermarking for AI-Generated Text"](https://openreview.net/pdf?id=SsmT8aO45L), ICLR 2024.
- `dipmark`: Wu et al., ["A Resilient and Accessible Distribution-Preserving Watermark for Large Language Models"](https://openreview.net/pdf?id=c8qWiNiqRY), ICML 2024.
- `hcw`: Hu et al., ["Unbiased Watermark for Large Language Models"](https://openreview.net/forum?id=uWVC5FVidc), ICLR 2024.
- `heavywater`: Tsur et al., ["HeavyWater and SimplexWater: Distortion-free LLM Watermarks for Low-Entropy Distributions"](https://openreview.net/forum?id=R5EBtNE2Y9), NeurIPS 2025.
- `simplexwater`: Tsur et al., ["HeavyWater and SimplexWater: Distortion-free LLM Watermarks for Low-Entropy Distributions"](https://openreview.net/forum?id=R5EBtNE2Y9), NeurIPS 2025.
- `kuditipudi`: Kuditipudi et al., ["Robust Distortion-free Watermarks for Language Models"](https://openreview.net/forum?id=FpaCL1MO2C), TMLR 2024.
- `semstamp`: Hou et al., ["SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation"](https://aclanthology.org/2024.naacl-long.226/), NAACL 2024.
- `pmark`: Huo et al., ["PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints"](https://arxiv.org/abs/2509.21057), 2025.
- `simmark`: Dabiriaghdam and Wang, ["SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models"](https://arxiv.org/pdf/2502.02787), 2025.
- `cgw`: Christ, Gunn, and Zamir, ["Undetectable Watermarks for Language Models"](https://proceedings.mlr.press/v247/christ24a.html), COLT 2024.
- `gaussmark`: Block, Rakhlin, and Sekhari, ["GaussMark: A Practical Approach for Structural Watermarking of Language Models"](https://openreview.net/pdf?id=YG3DbpAQBf), ICML 2025.
- `dawa`: He et al., ["Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach"](https://openreview.net/forum?id=Lzi8raVEQu), 2025.
- `hybrid`: ["Catch-22: On the Fundamental Tradeoff Between Detectability and Robustness in LLM Watermarking"](https://icml.cc/virtual/2026/poster/66807), ICML 2026.

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

The full pipeline handles the `vanilla` baseline automatically. For AUROC in individual-stage runs, generate and attack the `vanilla` baseline before evaluation, then score it with the selected detector:

```bash
python -m catch22.generate --config configs/llama2_lfqa.yaml --method vanilla --resume
python -m catch22.attack --config configs/llama2_lfqa.yaml --method vanilla --attack dipper --paraphrase-strength moderate --resume
python -m catch22.attack --config configs/llama2_lfqa.yaml --method vanilla --attack extreme-paraphrase --paraphrase-strength extreme --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --source-method vanilla --condition clean --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --source-method vanilla --condition dipper_moderate --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --source-method vanilla --condition extreme_paraphrase --resume
python -m catch22.evaluate --config configs/llama2_lfqa.yaml --method hybrid
```

Use `--dry-run` on any command to print resolved paths and planned work without running models.

## Outputs

Generated outputs are written under `outputs/` and are ignored by git:

```text
outputs/<track>/<method>/clean/generations.jsonl
outputs/<track>/<method>/attacks/<condition>/attacked.jsonl
outputs/<track>/<method>/scored/<condition>/scored.jsonl
outputs/<track>/<method>/scored_baseline/vanilla/<condition>/scored.jsonl
outputs/<track>/<method>/evaluations/<condition>.json
outputs/<track>/tables/*.json
outputs/<track>/tables/*.tex
```
