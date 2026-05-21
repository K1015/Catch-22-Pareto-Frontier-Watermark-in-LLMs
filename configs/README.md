# Configs

This directory contains YAML configs for LFQA reproduction runs.

- `llama2_lfqa.yaml`: Llama2 setup using `meta-llama/Llama-2-7b-hf`.
- `mistral_lfqa.yaml`: Mistral setup using `mistralai/Mistral-7B-v0.1`.

Validate the configs before running experiments:

```bash
python -m catch22.validate_config configs/llama2_lfqa.yaml
python -m catch22.validate_config configs/mistral_lfqa.yaml
```

Run the full configured suite:

```bash
python -m catch22.pipeline --config configs/llama2_lfqa.yaml --reproduction-suite --resume
python -m catch22.pipeline --config configs/mistral_lfqa.yaml --reproduction-suite --resume
```

Use `--model-name-or-path` with `catch22.generate` or `catch22.pipeline` to point at a local model checkpoint.

## HPC run files

Create scheduler files outside the tracked source tree, or use names ending in `.sbatch` or `.slurm` so they stay ignored by git. A GPU run file should load your Python environment, change to the repository root, and call the same module commands used locally:

```bash
#!/bin/bash
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH --time=<time_limit>
#SBATCH --cpus-per-task=<cpus>
#SBATCH --mem=<memory>

set -euo pipefail
cd <repo_root>
source <venv_path>/bin/activate
python -m catch22.pipeline --config configs/llama2_lfqa.yaml --reproduction-suite --resume
```

Replace the angle-bracket values with the settings for your cluster.
