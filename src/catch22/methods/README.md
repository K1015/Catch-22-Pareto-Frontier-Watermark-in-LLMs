# Watermark Methods

Each watermark has a dedicated module in this folder. The common interface is defined in `base.py`, and `__init__.py` maps method names to implementation classes.

Method files:

- `kgw.py`: context-dependent green-list watermark.
- `unigram.py`: fixed-green-list unigram watermark.
- `dipmark.py`: distribution-preserving p-alpha reweighting.
- `hcw.py`: Hu-style bias-free inverse-CDF reweighting.
- `heavywater.py`: heavy-tailed PRF token preference watermark.
- `simplexwater.py`: simplex-style PRF direction watermark.
- `kuditipudi.py`: Gumbel-key randomized sampling watermark.
- `semantic.py`: `semstamp`, `pmark`, and `simmark` semantic text watermarks.
- `cgw.py`: Christ-style distribution-preserving inverse-CDF sampling.
- `gaussmark.py`: inference hook for evaluating GaussMark checkpoints through the common CLI.
- `dawa.py`: distribution-aware adaptive watermark.
- `hybrid.py`: token-level plus semantic hybrid watermark.
- `vanilla.py`: unwatermarked baseline.

Paper links:

- `kgw.py`: Kirchenbauer et al., ["A Watermark for Large Language Models"](https://openreview.net/pdf?id=aX8ig9X2a7), ICML 2023.
- `unigram.py`: Zhao et al., ["Provable Robust Watermarking for AI-Generated Text"](https://openreview.net/pdf?id=SsmT8aO45L), ICLR 2024.
- `dipmark.py`: Wu et al., ["A Resilient and Accessible Distribution-Preserving Watermark for Large Language Models"](https://openreview.net/pdf?id=c8qWiNiqRY), ICML 2024.
- `hcw.py`: Hu et al., ["Unbiased Watermark for Large Language Models"](https://openreview.net/forum?id=uWVC5FVidc), ICLR 2024.
- `heavywater.py`: Tsur et al., ["HeavyWater and SimplexWater: Distortion-free LLM Watermarks for Low-Entropy Distributions"](https://openreview.net/forum?id=R5EBtNE2Y9), NeurIPS 2025.
- `simplexwater.py`: Tsur et al., ["HeavyWater and SimplexWater: Distortion-free LLM Watermarks for Low-Entropy Distributions"](https://openreview.net/forum?id=R5EBtNE2Y9), NeurIPS 2025.
- `kuditipudi.py`: Kuditipudi et al., ["Robust Distortion-free Watermarks for Language Models"](https://openreview.net/forum?id=FpaCL1MO2C), TMLR 2024.
- `semantic.py` / `semstamp`: Hou et al., ["SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation"](https://aclanthology.org/2024.naacl-long.226/), NAACL 2024.
- `semantic.py` / `pmark`: Huo et al., ["PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints"](https://arxiv.org/abs/2509.21057), 2025.
- `semantic.py` / `simmark`: Dabiriaghdam and Wang, ["SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models"](https://arxiv.org/pdf/2502.02787), 2025.
- `cgw.py`: Christ, Gunn, and Zamir, ["Undetectable Watermarks for Language Models"](https://proceedings.mlr.press/v247/christ24a.html), COLT 2024.
- `gaussmark.py`: Block, Rakhlin, and Sekhari, ["GaussMark: A Practical Approach for Structural Watermarking of Language Models"](https://openreview.net/pdf?id=YG3DbpAQBf), ICML 2025.
- `dawa.py`: He et al., ["Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach"](https://openreview.net/forum?id=Lzi8raVEQu), 2025.
- `hybrid.py`: ["Catch-22: On the Fundamental Tradeoff Between Detectability and Robustness in LLM Watermarking"](https://icml.cc/virtual/2026/poster/66807), ICML 2026.

Run one method from the repository root:

```bash
python -m catch22.generate --config configs/llama2_lfqa.yaml --method kgw --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method kgw --condition clean --resume
```

For HPC execution, create an untracked scheduler file in your own workspace. The file should activate the Python environment, change to the repository root, and run the same `python -m catch22...` commands. Do not commit scheduler files or machine-specific paths.
