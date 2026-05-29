# Catch-22: Pareto Frontier for Detectability and Robustness in LLM Watermarking

LLM watermarking has its own Catch-22[^catch22-name]: watermarks that are easy to verify are often easier to notice, while watermarks that stay hidden are easier to remove with edits.

[^catch22-name]: The name alludes to Joseph Heller's *Catch-22*, a paradoxical dilemma in which one decision cannot be made without negating another. In the context of LLMs, watermarks face an analogous bind: improving robustness often makes them more detectable, while reducing detectability weakens their robustness.

This repository is the reproduction package for the accepted ICML 2026 paper "Catch-22: On the Fundamental Tradeoff Between Detectability and Robustness in LLM Watermarking" by Kuheli Pratihar and Debdeep Mukhopadhyay.

The experiments focus on Long-Form Question Answering (LFQA), where a model writes detailed answers to open-ended questions. The supported model setups are:

- `meta-llama/Llama-2-7b-hf`
- `mistralai/Mistral-7B-v0.1`

The experiments evaluate every implemented watermark method, including the `hybrid` method, on clean outputs and under seven attack conditions (no attack, DIPPER, OPT-2.7B, watermark-removal, synonym substitution, back-translation, and summarization). Each condition reports keyed robustness (AUROC and TPR at 1% FPR) and keyless detectability (z-score) against unwatermarked `vanilla` outputs.

All paths, model identifiers, and cluster settings are supplied through `CATCH22_*` environment macros (see [Configuration](#configuration)); the repository hard-codes no user name, cluster name, or absolute path.

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

## Repository layout

| Path | What it is |
| --- | --- |
| `env.example.sh` | Template for all `CATCH22_*` path/site macros. Copy to `env.sh` and edit. |
| `common/` | Shared pipeline: method & attack registries, generation, attacks, scoring, evaluation, table rendering, and SLURM launchers. |
| `Llama2-Watermark/` | Per-method inference wrappers (model-agnostic; used for both Llama-2 and Mistral). |
| `experiments/llama2_lfqa_main/` | Llama-2-7B main experiment (paper Table 1): manifest + submit script. |
| `experiments/mistral7b_appendix/` | Mistral-7B appendix experiment: manifest + submit script. |
| `external/` | Upstream method repos (SemStamp, PMark, HeavyWater/SimplexWater), fetched by `scripts/fetch_external.sh`. |
| `scripts/` | `setup_env.sh`, `fetch_external.sh`, `sync_to_cluster.sh`. |
| `data/lfqa/inputs.jsonl` | LFQA prompts used by both tracks (500 sampled per run). |
| `docs/`, `figures/` | Run guides and static figures. |

## Methods and Conditions

| Method | Code | Reference |
| --- | --- | --- |
| `kgw` | `Llama2-Watermark/llama2_KGW_inference_LFQA.py` | Kirchenbauer et al., ["A Watermark for Large Language Models"](https://openreview.net/pdf?id=aX8ig9X2a7), ICML 2023 |
| `unigram` | `Llama2-Watermark/llama2_Unigram_inference_LFQA.py` | Zhao et al., ["Provable Robust Watermarking for AI-Generated Text"](https://openreview.net/pdf?id=SsmT8aO45L), ICLR 2024 |
| `dipmark` | `Llama2-Watermark/llama2_DiPMark_inference_LFQA.py` | Wu et al., ["A Resilient and Accessible Distribution-Preserving Watermark for Large Language Models"](https://openreview.net/pdf?id=c8qWiNiqRY), ICML 2024 |
| `hcw` | `Llama2-Watermark/llama2_HCW_inference_LFQA.py` | Hu et al., ["Unbiased Watermark for Large Language Models"](https://openreview.net/forum?id=uWVC5FVidc), ICLR 2024 |
| `heavywater` | `Llama2-Watermark/llama2_HeavyWater_inference_LFQA.py` | Tsur et al., ["HeavyWater and SimplexWater: Distortion-free LLM Watermarks for Low-Entropy Distributions"](https://openreview.net/forum?id=R5EBtNE2Y9), NeurIPS 2025 |
| `simplexwater` | `Llama2-Watermark/llama2_SimplexWater_inference_LFQA.py` | Tsur et al., ["HeavyWater and SimplexWater: Distortion-free LLM Watermarks for Low-Entropy Distributions"](https://openreview.net/forum?id=R5EBtNE2Y9), NeurIPS 2025 |
| `kuditipudi` | `Llama2-Watermark/llama2_Kuditipudi_inference_LFQA.py` | Kuditipudi et al., ["Robust Distortion-free Watermarks for Language Models"](https://openreview.net/forum?id=FpaCL1MO2C), TMLR 2024 |
| `semstamp` | `Llama2-Watermark/llama2_SemStamp_inference_LFQA.py` | Hou et al., ["SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation"](https://aclanthology.org/2024.naacl-long.226/), NAACL 2024 |
| `pmark` | `Llama2-Watermark/llama2_PMark_inference_LFQA.py` | Huo et al., ["PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints"](https://arxiv.org/abs/2509.21057), 2025 |
| `simmark` | `Llama2-Watermark/llama2_SimMark_inference_LFQA.py` | Dabiriaghdam and Wang, ["SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models"](https://arxiv.org/pdf/2502.02787), 2025 |
| `cgw` | `Llama2-Watermark/llama2_CGW_inference_LFQA.py` | Christ, Gunn, and Zamir, ["Undetectable Watermarks for Language Models"](https://proceedings.mlr.press/v247/christ24a.html), COLT 2024 |
| `gaussmark` | `Llama2-Watermark/llama2_GaussMark_inference_LFQA.py` | Block, Rakhlin, and Sekhari, ["GaussMark: A Practical Approach for Structural Watermarking of Language Models"](https://openreview.net/pdf?id=YG3DbpAQBf), ICML 2025 |
| `dawa` | `common/dawa_watermark.py` | He et al., ["Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach"](https://openreview.net/forum?id=Lzi8raVEQu), 2025 |
| `hybrid` | `Llama2-Watermark/llama2_Hybrid_inference_LFQA.py` | ["Catch-22: On the Fundamental Tradeoff Between Detectability and Robustness in LLM Watermarking"](https://icml.cc/virtual/2026/poster/66807), ICML 2026 |

The semantic (`semstamp`, `pmark`, `simmark`) and distortion-free (`heavywater`, `simplexwater`) methods build on the upstream repositories fetched into `external/`. Attack conditions (defined in `common/attack_registry.py`):

- `none`: no attack (clean baseline).
- `dipper`: DIPPER paraphrase ($\hat\varepsilon \approx 0.25$).
- `opt`: OPT-2.7B paraphrase ($\hat\varepsilon \approx 0.15$).
- `wm-removal`: watermark-removal prompting ($\hat\varepsilon \approx 0.15$).
- `synonym`: WordNet synonym substitution ($\hat\varepsilon \approx 0.15$).
- `backtranslation`: round-trip translation ($\hat\varepsilon \approx 0.42$).
- `summarization`: summary-style rewrite ($\hat\varepsilon \approx 0.55$).

## Installation

```bash
# 1. create and activate a Python 3.10+ environment
conda create -y -n catch22 python=3.10 && conda activate catch22

# 2. configure paths
cp env.example.sh env.sh && source env.sh   # edit env.sh first

# 3. install deps, NLTK corpora, and clone the upstream method repos
scripts/setup_env.sh
```

`scripts/setup_env.sh` installs `requirements.txt`, the NLTK corpora used by the attacks, and runs `scripts/fetch_external.sh` (which clones SemStamp, PMark, and HeavyWater/SimplexWater at pinned commits into `external/`). For gated checkpoints (Llama-2), run `huggingface-cli login` first, or point `CATCH22_MODEL_*` at local snapshots under `CATCH22_HF_CACHE`.

## Configuration

All configuration is provided through `CATCH22_*` environment variables. Copy the template and set the values for the target machine or cluster:

```bash
cp env.example.sh env.sh
# edit env.sh: CATCH22_ROOT (auto-detected), CATCH22_HF_CACHE, CATCH22_CONDA_ENV,
# CATCH22_MODEL_LLAMA2 / CATCH22_MODEL_MISTRAL, and the SLURM site variables.
source env.sh
```

`env.sh` is git-ignored, so local paths are not committed. Key macros: `CATCH22_ROOT`, `CATCH22_DATA_DIR`, `CATCH22_RESULTS_DIR`, `CATCH22_HF_CACHE`, `CATCH22_CONDA_ENV`, `CATCH22_MODEL_LLAMA2` / `CATCH22_MODEL_MISTRAL`, and the SLURM knobs `CATCH22_SLURM_GPU_PARTITION` / `CATCH22_SLURM_CPU_PARTITION` / `CATCH22_SLURM_ACCOUNT` and the queue caps. See `env.example.sh` for the full list and defaults.

## Data

The expected LFQA input file is:

```text
data/lfqa/inputs.jsonl
```

The repository includes the LFQA prompts used for the experiments (500 sampled per run). Each row must include a `prompt` field and may include `id` and `reference`. To use your own prompts, set `dataset_path` in the manifest or point `CATCH22_DATA_DIR` at another location.

## Reproducing the experiments

Each experiment is a manifest describing methods, attacks, model, and sampling parameters. Validate it first:

```bash
source env.sh
python common/validate_experiment.py --manifest experiments/llama2_lfqa_main/manifest.json
python common/validate_experiment.py --manifest experiments/mistral7b_appendix/manifest.json
```

Then submit the whole pipeline — asset staging → preflight → clean generation → attacks → scoring → keyed/keyless evaluation → table render, for every method × attack, with SLURM job dependencies — on a cluster:

```bash
experiments/llama2_lfqa_main/slurm/submit_all.sh      # Llama-2-7B main table
experiments/mistral7b_appendix/slurm/submit_all.sh    # Mistral-7B appendix table
```

Rendered tables are written to `$CATCH22_RESULTS_DIR/<track>/rendered/`. For a single-GPU run without SLURM, see the stage-by-stage commands in [`docs/reproduce_lfqa.md`](docs/reproduce_lfqa.md), and [`docs/slurm.md`](docs/slurm.md) for the launcher/macro reference.

## Individual stages

The pipeline stages are plain Python modules under `common/` (run from the repository root after `source env.sh`):

```bash
python common/run_generation.py   --manifest <manifest> --method <method> --output-file <...> --summary-file <...> --resume
python common/run_attack_suite.py --manifest <manifest> --attack-name <attack> --input-file <...> --output-file <...> --summary-file <...>
python common/score_outputs.py    --manifest <manifest> --method <method> --input-file <...> --output-file <...> --summary-file <...>
python common/evaluate_outputs.py --positive-file <...> --negative-file <...> --keyless-calibration-file <...> --tokenizer-model <model> --output-json <...> --target-fpr 0.01
python common/render_tables.py    --manifest <manifest> --results-root <...>
```

See [`docs/reproduce_lfqa.md`](docs/reproduce_lfqa.md) for a fully worked example.

## Outputs

Run artifacts are written under `$CATCH22_RESULTS_DIR/<track>/` (git-ignored):

```text
<track>/<method>/raw/clean.jsonl
<track>/<method>/attacks/<attack>/{positive,negative}_attacked.jsonl
<track>/<method>/scored/<attack>/{positive,negative}_scored.jsonl
<track>/<method>/evaluations/<attack>.json
<track>/rendered/*.tex
<track>/rendered/*.json
```
