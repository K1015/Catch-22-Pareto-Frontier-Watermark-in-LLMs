# Additional Rebuttal Results

This repository contains all additional results requested by reviewers during the rebuttal phase.

## Per-Token Change Probabilities Under Different Edit Channels

To assess the validity of the i.i.d. edit channel assumption underlying Theorem 2, we conducted a token-level analysis of representative paraphrasing attacks. For each attack type, we fix ten 100-token watermarked Llama-2 outputs, apply the attack 10 times, and for every token index $t$, estimate the empirical probability that the token at position $t$ remains unchanged (blue) or is modified (orange) across iterations. The horizontal dashed line marks the global average edit rate $\varepsilon$ for that attack.

### DIPPER Paraphrasing Attack

DIPPER paraphrasing attack (average edit rate $\varepsilon \approx 0.25$).

![DIPPER Probability Distribution](DIPPER-prob.png)

DIPPER produces a nearly uniform edit profile across all token positions: each position is modified with probability close to $\varepsilon = 0.25$, exhibiting only minor fluctuations. This behavior closely matches the i.i.d. substitution channel assumed in our theoretical analysis, validating Theorem 2 as an appropriate model for this class of paraphrasing attacks.

### Synonym Substitution Attack

Synonym-substitution attack (lexical substitution calibrated to $\varepsilon \approx 0.15$).

![Synonym Substitution Probability Distribution](SynonymSubstitution-prob.png)

Synonym substitution exhibits somewhat greater positional variability than DIPPER, with a slight increase in modification probability toward the second half of the sequence. Nevertheless, the per-position edit probability remains concentrated around the global rate $\varepsilon \approx 0.15$, indicating that our i.i.d. model provides a reasonable first-order approximation for this attack as well.

### Back-Translation Attack

Back-translation attack (en→fr→en).

![Back-Translation Probability Distribution](BackTranslation-prob.png)

Back-translation produces localized spikes of correlated edits due to unconstrained semantic rewriting, with no systematic positional bias but substantially higher and more variable edit rates than the other attacks. This violation of the fixed-rate i.i.d. assumption corresponds to the high-noise regime where Theorem 2 predicts detection failure—consistent with our experimental observation that all watermarking schemes fail under back-translation (see Table 2 in the main paper).

## Summary

These measurements demonstrate that practical paraphrasers are not strictly i.i.d., as they introduce short correlated spans of edits. However, marginally over positions, the edit probability is approximately uniform and tightly concentrated around $\varepsilon$ for attacks like DIPPER and synonym substitution. Consequently, our first-order i.i.d. substitution channel model in Theorem 2 provides a valid approximation for these realistic attacks, with higher-order semantic dependencies manifesting only as small deviations around the global edit rate. When paraphrasing operates with unconstrained edits, as in back-translation, all watermarking schemes fail—consistent with both our theoretical predictions and experimental observations.

## Paraphrasing Datasets

The paraphrasing datasets used for i.i.d. channel edit model validation experiments are available as JSON files:

- `synonym_substitution_paraphrase_dataset.json`
- `dipper_paraphrase_dataset.json`
- `backtranslation_paraphrase_dataset.json`

## Robustness-Detectability Trade-off

Detailed results on the robustness-detectability trade-off across different watermarking schemes are available on the [Performance Evaluation](Performance-evaluation.md) page.
