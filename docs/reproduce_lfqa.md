# Reproducing the LFQA experiments

This guide assumes the package was installed with `pip install -e .`.

## Validate configs

```bash
python -m catch22.validate_config configs/llama2_lfqa.yaml
python -m catch22.validate_config configs/mistral_lfqa.yaml
```

## Reproduction suite runs

Llama2:

```bash
python -m catch22.pipeline --config configs/llama2_lfqa.yaml --reproduction-suite --resume
```

Mistral:

```bash
python -m catch22.pipeline --config configs/mistral_lfqa.yaml --reproduction-suite --resume
```

The pipeline runs clean generation, moderate Dipper paraphrasing, extreme summarization-style paraphrasing, scoring, evaluation, and table rendering for every configured watermark method. It also generates and attacks an unwatermarked `vanilla` baseline, then scores that baseline with each detector so AUROC can be computed for every condition.

## Selected method run

```bash
python -m catch22.generate --config configs/llama2_lfqa.yaml --method hybrid --resume
python -m catch22.attack --config configs/llama2_lfqa.yaml --method hybrid --attack dipper --paraphrase-strength moderate --resume
python -m catch22.attack --config configs/llama2_lfqa.yaml --method hybrid --attack extreme-paraphrase --paraphrase-strength extreme --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --condition clean --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --condition dipper_moderate --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --condition extreme_paraphrase --resume
python -m catch22.evaluate --config configs/llama2_lfqa.yaml --method hybrid
python -m catch22.render --config configs/llama2_lfqa.yaml
```

To compute AUROC in a selected-method run, run these baseline commands before `evaluate` and `render`:

```bash
python -m catch22.generate --config configs/llama2_lfqa.yaml --method vanilla --resume
python -m catch22.attack --config configs/llama2_lfqa.yaml --method vanilla --attack dipper --paraphrase-strength moderate --resume
python -m catch22.attack --config configs/llama2_lfqa.yaml --method vanilla --attack extreme-paraphrase --paraphrase-strength extreme --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --source-method vanilla --condition clean --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --source-method vanilla --condition dipper_moderate --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --source-method vanilla --condition extreme_paraphrase --resume
python -m catch22.evaluate --config configs/llama2_lfqa.yaml --method hybrid
python -m catch22.render --config configs/llama2_lfqa.yaml
```

## Environment-check mode

Use the local backend for CI and local environment checks:

```bash
python -m catch22.pipeline --config configs/llama2_lfqa.yaml --reproduction-suite --num-samples 2 --local-backend --resume
python -m catch22.pipeline --config configs/mistral_lfqa.yaml --reproduction-suite --num-samples 2 --local-backend --resume
```
