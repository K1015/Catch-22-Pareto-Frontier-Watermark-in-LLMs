# Catch-22: Pareto Frontier for Detectability and Robustness in LLM Watermarking

This repository contains the code to replicate the results of the manuscript submission to ICML 2026 titled - "Catch-22: Pareto Frontier for Detectability and Robustness in LLM Watermarking".

## Overview

Large language models generate text through probabilistic token sampling, a mechanism increasingly leveraged for inference-time watermarking to verify AI-generated content. We present an information-theoretic framework that characterizes the trade-off between robustness to text editing and detectability by keyless observers, where detectability bounds are information-theoretic and computational attainability depends on detector access. Central to our analysis is an additive, usable Kullback-Leibler (KL) information budget that governs hypothesis testing separability between watermarked and unwatermarked outputs subject to a stealth constraint. This budget induces a hierarchy of detectability across watermark families: distribution-preserving schemes exhibit zero statistical drift, while probability-modifying schemes at both token and sentence levels accumulate detectable signal with sequence length. When text editing is modeled as a noise process, the usable KL budget contracts quadratically with edit rate for token-level schemes and according to an induced semantic flip rate for sentence-level schemes. These contraction laws reveal an irreducible trilemma among robustness, stealth, and reliable verification. Guided by these limits, we propose a hybrid watermarking strategy that selects among distribution-preserving, semantic-level, and token-level methods based on anticipated editing regimes. Experiments on Llama-2-7B and Mistral-7B under paraphrasing attacks corroborate theoretical predictions and confirm that the hybrid strategy is empirically near-Pareto across evaluated edit regimes.

![Watermarking schemes in modern LLMs exhibit a trade-off between detectability via statistical tests and robustness against LLM output editing.](figures/HLV3.png)
Figure: Watermarking schemes in modern LLMs exhibit a trade-off between detectability via statistical tests and robustness against LLM output editing.

## Installation

### Setting up Conda Environment

We recommend using a conda environment for reproducibility:

```bash
# Create a new conda environment
conda create -n llm-watermark python=3.10
conda activate llm-watermark

# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

## Model Access

### Accessing Llama 2-7B Model

This work uses the Llama 2-7B model, which requires:
- **Minimum GPU Memory**: 16GB VRAM (with 8-bit quantization) or 28GB VRAM (full precision)
- **Recommended**: NVIDIA RTX A4000 (16GB) or better for quantized inference
- **Optimal**: NVIDIA A100 (40GB) or H100 for full precision and faster generation

### Accessing Llama 2 Models

To use Llama 2 models, you need to request access from Meta and authenticate with Hugging Face:

1. **Request Access from Meta**:
   - Visit [Meta's Llama 2 page](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
   - Fill out the access request form
   - Wait for approval (typically 24-48 hours)

2. **Link to Hugging Face**:
   - Create a [Hugging Face account](https://huggingface.co/join) if you don't have one
   - Visit the [Llama 2 model page](https://huggingface.co/meta-llama/Llama-2-7b-hf)
   - Request access using the same email as your Meta request
   - Once approved, you'll receive confirmation

3. **Authenticate Locally**:
   ```bash
   huggingface-cli login
   # Enter your Hugging Face access token when prompted
   ```

4. **Verify Access**:
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
   ```

## Data Preparation

First, download the LFQA dataset by running the following command:

```bash
cd Llama2-Watermark/data
bash download_data.sh
```
This will download and prepare the Long-Form Question Answering (LFQA) dataset.

## Generating Watermarked Text

### Basic Usage

You can generate watermarked text by running the following command:

```bash
python llama2_KGW_inference_LFQA.py
```

### Available Watermarking Schemes

Generate text with different watermarking methods using:

```bash
python llama2_XX_inference_LFQA.py
```

where `XX` is the watermark type:
- `KGW` - [Kirchenbauer et al.](https://arxiv.org/abs/2301.10226) (Biased sampling)
- `Unigram` - [Zhao et al.](https://arxiv.org/abs/2306.17439) (Biased sampling)
- `HCW` - [Hu et al.](https://arxiv.org/abs/2310.10669) (Bias-free sampling)
- `DiPMark` - [Wu et al.](https://arxiv.org/abs/2310.07710) (Bias-free sampling)
- `CGW` - [Christ et al.](https://proceedings.mlr.press/v247/christ24a/christ24a.pdf) (Distribution-preserving)
- `Hybrid` - Our proposed adaptive hybrid scheme (Theorem 3 in the manuscript)
- `vanilla` - Non-watermarked baseline

## Visualization and Analysis

### Generating Figures

The `Analysis-Results/` directory contains plotting scripts and data for reproducing paper figures:

```bash
cd Analysis-Results/

# Generate Figure A: Detectability scaling
python Fig1Plot.py

# Generate Figure B: Robustness degradation  
python Fig2Plot.py

# Generate Figure C: Pareto frontier
python Fig3Plot.py
```

The directory also includes:
- Pre-generated scatter plot data (`.npz` files)
- Annotated figure PDFs explaining key results
- Generated plots (`.png` files)
---

*Note: This is work under submission. Citations and license information will be added upon acceptance.*

