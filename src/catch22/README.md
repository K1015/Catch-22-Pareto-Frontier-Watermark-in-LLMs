# Catch-22 Package

This package exposes the experiment stages as Python modules:

- `generate`: create clean LFQA generations for a selected watermark method.
- `attack`: apply moderate Dipper paraphrasing or extreme paraphrasing.
- `score`: run the detector on clean or attacked outputs.
- `evaluate`: compute detectability and robustness metrics.
- `render`: produce JSON and TeX summary tables.
- `pipeline`: run the configured suite across all methods.

Common commands:

```bash
python -m catch22.generate --config configs/llama2_lfqa.yaml --method hybrid --resume
python -m catch22.attack --config configs/llama2_lfqa.yaml --method hybrid --attack dipper --paraphrase-strength moderate --resume
python -m catch22.score --config configs/llama2_lfqa.yaml --method hybrid --condition clean --resume
python -m catch22.evaluate --config configs/llama2_lfqa.yaml --method hybrid
python -m catch22.render --config configs/llama2_lfqa.yaml
```

Use `--dry-run` to inspect resolved paths before running a stage.

For HPC execution, construct an untracked `.sbatch` or `.slurm` file that runs one of these module commands after activating the Python environment and changing to the repository root.
