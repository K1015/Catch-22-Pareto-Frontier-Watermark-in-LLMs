# LFQA input format

Place the LFQA prompts at `data/lfqa/inputs.jsonl`. Each row must be JSON with:

- `id`: stable sample identifier.
- `prompt`: full prompt passed to the model.
- `reference`: optional human/reference answer used only for record keeping.

The repository ships with three example rows for environment checks. Replace them with the paper LFQA prompts before a full reproduction run.

For HPC runs, make sure the scheduler job starts from the repository root before invoking Python:

```bash
cd <repo_root>
python -m catch22.validate_config configs/llama2_lfqa.yaml
```
