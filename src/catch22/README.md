# Catch-22 Package

This package exposes the experiment stages as Python modules:

- `generate`: create clean LFQA generations for a selected watermark method.
- `attack`: apply moderate Dipper paraphrasing or extreme paraphrasing.
- `score`: run the detector on clean, attacked, or baseline outputs.
- `evaluate`: compute detectability, AUROC, z-score, and robustness metrics.
- `render`: produce JSON and TeX summary tables with detectability and robustness metrics.
- `pipeline`: run the configured suite across all methods.

Watermark implementations live in `methods/`, with one module per method and a shared interface in `methods/base.py`.

Common commands:

```bash
python -m catch22.generate --config configs/llama2_lfqa.yaml --method hybrid --resume
python -m catch22.attack --config configs/llama2_lfqa.yaml --method hybrid --attack dipper --paraphrase-strength moderate --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --condition clean --resume
python -m catch22.evaluate --config configs/llama2_lfqa.yaml --method hybrid
python -m catch22.render --config configs/llama2_lfqa.yaml
```

Use `--source-method vanilla` with `score` when you want to score unwatermarked baseline outputs with a selected detector for AUROC:

```bash
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --source-method vanilla --condition clean --resume
```

Use `--dry-run` to inspect resolved paths before running a stage.

For HPC execution, construct an untracked `.sbatch` or `.slurm` file that runs one of these module commands after activating the Python environment and changing to the repository root.
