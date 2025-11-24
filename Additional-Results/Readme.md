This repository contains all additional results requested by Reviewers during the Rebuttal phase.

## Performance Comparison on Llama-2-7B under GPT-3.5 Paraphrasing Attacks

Performance of Biased (KGW, Unigram), Bias-free (DiPMark, HCW), Distribution-preserving (CGW), and Optimal Hybrid watermarking schemes on Llama-2-7B under three GPT-3.5 paraphrasing attacks at 15% edit rate.

- **Robustness**: AUROC, TPR at 1% FPR, F1 at 1% FPR
- **Detectability**: z-score (black-box, no key)

| Model | Attack | Method | AUROC | TPR@1% | F1@1% | z-score |
|:------|:-------|:-------|:-----:|:------:|:-----:|:-------:|
| **Llama-2-7B** | GPT-3.5 Synonym-based (avg ε≈0.15) | KGW (Biased) | 0.780 | 0.590 | 0.720 | 8.3 |
| | | Unigram (Biased) | 0.790 | 0.615 | 0.740 | 7.8 |
| | | DiPMark (Bias-free) | 0.905 | 0.855 | 0.900 | 3.5 |
| | | HCW (Bias-free) | 0.920 | 0.880 | 0.915 | 3.0 |
| | | CGW (Dist-pres.) | 0.502 | 0.310 | 0.420 | -5.5 |
| | | **Optimal Hybrid** | **0.930** | **0.895** | **0.922** | **4.4** |
| | GPT-3.5 Adversarial prompting ("remove watermark", avg ε≈0.15) | KGW (Biased) | 0.775 | 0.585 | 0.715 | 8.1 |
| | | Unigram (Biased) | 0.785 | 0.610 | 0.735 | 7.6 |
| | | DiPMark (Bias-free) | 0.900 | 0.850 | 0.895 | 3.4 |
| | | HCW (Bias-free) | 0.918 | 0.878 | 0.912 | 3.0 |
| | | CGW (Dist-pres.) | 0.501 | 0.305 | 0.415 | -5.6 |
| | | **Optimal Hybrid** | **0.928** | **0.892** | **0.919** | **4.3** |
| | GPT-3.5 Back-translation (en→fr→en) | KGW (Biased) | 0.580 | 0.520 | 0.560 | -1.2 |
| | | Unigram (Biased) | 0.570 | 0.505 | 0.552 | -1.0 |
| | | DiPMark (Bias-free) | 0.600 | 0.540 | 0.575 | -0.8 |
| | | HCW (Bias-free) | 0.590 | 0.530 | 0.568 | -0.6 |
| | | CGW (Dist-pres.) | 0.505 | 0.210 | 0.350 | -6.5 |
| | | **Optimal Hybrid** | **0.605** | **0.548** | **0.582** | **-0.5** |
