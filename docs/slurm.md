# Generic SLURM usage

This repository does not track scheduler scripts. Create `.sbatch` or `.slurm` files locally, keep them out of version control, and submit them with `sbatch <file>`.

Each run file should:

1. Request the partition, time, memory, CPU count, and GPU count required by the stage.
2. Change to the repository root with `cd <repo_root>`.
3. Activate the Python environment that has this package installed.
4. Run a `python -m catch22...` command.

GPU pipeline file:

```bash
#!/bin/bash
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=<cpus>
#SBATCH --mem=<memory>
#SBATCH --time=<time_limit>
#SBATCH --job-name=catch22-pipeline

set -euo pipefail
cd <repo_root>
source <venv_path>/bin/activate
python -m catch22.pipeline --config configs/llama2_lfqa.yaml --reproduction-suite --resume
```

CPU render/evaluation file:

```bash
#!/bin/bash
#SBATCH --partition=<cpu_partition>
#SBATCH --cpus-per-task=<cpus>
#SBATCH --mem=<memory>
#SBATCH --time=<time_limit>
#SBATCH --job-name=catch22-render

set -euo pipefail
cd <repo_root>
source <venv_path>/bin/activate
python -m catch22.evaluate --config configs/llama2_lfqa.yaml
python -m catch22.render --config configs/llama2_lfqa.yaml
```

Example GPU generation job:

```bash
sbatch \
  --partition=<gpu_partition> \
  --gres=gpu:1 \
  --cpus-per-task=<cpus> \
  --mem=<memory> \
  --time=<time_limit> \
  --job-name=catch22-generate \
  --wrap="python -m catch22.generate --config configs/llama2_lfqa.yaml --method hybrid --resume"
```

Example moderate paraphrase job:

```bash
sbatch \
  --partition=<gpu_partition> \
  --gres=gpu:1 \
  --cpus-per-task=<cpus> \
  --mem=<memory> \
  --time=<time_limit> \
  --job-name=catch22-dipper \
  --wrap="python -m catch22.attack --config configs/llama2_lfqa.yaml --method hybrid --attack dipper --paraphrase-strength moderate --resume"
```

Example CPU evaluation job:

```bash
sbatch \
  --partition=<cpu_partition> \
  --cpus-per-task=<cpus> \
  --mem=<memory> \
  --time=<time_limit> \
  --job-name=catch22-eval \
  --wrap="python -m catch22.evaluate --config configs/llama2_lfqa.yaml"
```

For arrays, replace `<array_limit>` with the concurrency allowed by your site:

```bash
sbatch \
  --array=0-13%<array_limit> \
  --partition=<gpu_partition> \
  --gres=gpu:1 \
  --time=<time_limit> \
  --wrap="python -m catch22.generate --config configs/llama2_lfqa.yaml --method \$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" methods.txt) --resume"
```
