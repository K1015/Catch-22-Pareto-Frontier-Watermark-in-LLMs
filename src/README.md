# Source

The Python package lives in `src/catch22`.

Install it in editable mode from the repository root:

```bash
pip install -e .
```

Then run modules with `python -m catch22.<stage>`, for example:

```bash
python -m catch22.generate --config configs/llama2_lfqa.yaml --method hybrid --dry-run
python -m catch22.pipeline --config configs/mistral_lfqa.yaml --reproduction-suite --dry-run
```

For HPC execution, put the same commands in an untracked `.sbatch` or `.slurm` file. The file should activate the environment, change to `<repo_root>`, and then call `python -m catch22...`.
