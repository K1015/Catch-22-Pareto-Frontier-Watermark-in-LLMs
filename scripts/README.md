# Scripts

This directory is reserved for optional local helpers. The reproduction workflow is exposed through `python -m catch22.*` entrypoints so that users can run the same commands locally or inside their own scheduler wrappers.

Primary commands:

```bash
python -m catch22.pipeline --config configs/llama2_lfqa.yaml --reproduction-suite --resume
python -m catch22.pipeline --config configs/mistral_lfqa.yaml --reproduction-suite --resume
```

## HPC run files

Do not add cluster-specific shell scripts to this directory. For HPC execution, create an untracked file such as `run_llama2.sbatch` from this pattern:

```bash
#!/bin/bash
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH --time=<time_limit>

set -euo pipefail
cd <repo_root>
source <venv_path>/bin/activate
python -m catch22.pipeline --config configs/llama2_lfqa.yaml --reproduction-suite --resume
```

Submit it with `sbatch run_llama2.sbatch`.
