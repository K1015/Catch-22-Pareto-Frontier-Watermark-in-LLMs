# Data

Place LFQA inputs under `data/lfqa/inputs.jsonl`.

The pipeline reads this path from the active config:

```bash
python -m catch22.validate_config configs/llama2_lfqa.yaml
python -m catch22.generate --config configs/llama2_lfqa.yaml --method hybrid --dry-run
```

Generated model outputs, attack outputs, scores, metrics, and tables are written to `outputs/`, which is ignored by git.

## HPC data placement

On a cluster, copy or mount the LFQA input so the active config can still resolve `data/lfqa/inputs.jsonl` from the repository root. If your dataset lives elsewhere, edit `dataset_path` in the YAML config or pass a config copy with the correct relative path.

Keep scheduler files untracked by naming them with `.sbatch` or `.slurm`, then submit them with `sbatch <file>`.
