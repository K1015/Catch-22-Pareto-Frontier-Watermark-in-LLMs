# Docs

This directory contains run guides for local and cluster environments.

- `reproduce_lfqa.md`: exact Llama2 and Mistral commands for generation, attacks, scoring, evaluation, and rendering.
- `slurm.md`: generic `sbatch --wrap` examples with replaceable values for site-specific partitions and limits.

Start with:

```bash
python -m catch22.pipeline --config configs/llama2_lfqa.yaml --reproduction-suite --dry-run
```

## HPC run files

Use `slurm.md` to construct local `.sbatch` or `.slurm` files. Keep those files out of version control; the repository ignores these suffixes. A run file should contain only generic cluster settings, environment activation, `cd <repo_root>`, and the relevant `python -m catch22...` command.
