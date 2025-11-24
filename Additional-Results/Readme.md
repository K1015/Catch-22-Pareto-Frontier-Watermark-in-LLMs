# Additional Rebuttal Results

This repository contains all additional results requested by Reviewers during the Rebuttal phase.

## Per-Token Change Probabilities Under Different Edit Channels

For each attack type, we fix ten single 100-token watermarked Llama-2 outputs, apply the attack 10 times, and for every token index *t*, estimate the empirical probability that the token at position *t* is left unchanged (blue) or modified (orange) across 10 iterations of the same prompt. The horizontal dashed line marks the global average edit rate ε for that attack.

### DIPPER Paraphrasing Attack
DIPPER paraphrasing attack (average edit rate ε ≈ 0.25).

![DIPPER Probability Distribution](DIPPER-prob.png)

### Synonym Substitution Attack
Synonym-substitution attack (lexical substitution calibrated to ε ≈ 0.15).

![Synonym Substitution Probability Distribution](SynonymSubstitution-prob.png)

### Back-Translation Attack
Back-translation attack (en→fr→en).

![Back-Translation Probability Distribution](BackTranslation-prob.png)

## Paraphrasing Datasets

The paraphrasing datasets used for i.i.d. channel edit model validation experiments are available as JSON files:

- `synonym_substitution_paraphrase_dataset.json`
- `dipper_paraphrase_dataset.json`
- `backtranslation_paraphrase_dataset.json`

## Robustness-Detectability Trade-off

Detailed results on the robustness-detectability trade-off across different watermarking schemes are available in the [Performance Evaluation](Performance-evaluation.md) page.
