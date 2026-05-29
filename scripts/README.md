# Scripts

Setup and transfer utilities for the Catch-22 reproduction package. Every path
is supplied through `CATCH22_*` macros defined in `env.sh` (copied from
`env.example.sh`); nothing here hard-codes a machine, cluster, or user account.
Each script re-sources `env.sh` (falling back to `env.example.sh`) when the
caller has not already done so, so the standard invocation is:

```bash
cp env.example.sh env.sh   # edit, then
source env.sh
scripts/<script>.sh
```

| Script | Purpose |
| --- | --- |
| `setup_env.sh` | Install the pipeline dependencies and NLTK corpora, then clone the upstream method repositories. |
| `fetch_external.sh` | Clone the upstream method repositories at pinned commits into `external/`. |
| `sync_to_cluster.sh` | Mirror the code and manifests to a remote cluster with `rsync`. |

## `setup_env.sh`

Run once inside the activated Python 3.10+ environment (the conda environment or
venv named by `CATCH22_CONDA_ENV`). In order, it:

1. installs the pipeline dependencies from `requirements.txt`;
2. downloads the NLTK corpora required by the attack suite (`punkt`,
   `punkt_tab`, `wordnet`, `omw-1.4`);
3. runs `fetch_external.sh` to clone the upstream method repositories.

```bash
source env.sh
scripts/setup_env.sh
```

The script installs dependencies from `requirements.txt` alone; there is no
editable package installation. The reproduction pipeline resides in `common/`
and `Llama2-Watermark/`, and is driven through the manifests in
`experiments/llama2_lfqa_main/` and `experiments/mistral7b_appendix/`. On
completion, the script prints a `common/validate_experiment.py` invocation for
validating a manifest. For gated checkpoints (Llama-2), run
`huggingface-cli login` first, or point `CATCH22_MODEL_*` at local snapshots
under `CATCH22_HF_CACHE`.

## `fetch_external.sh`

Clones the upstream watermark method repositories into `$CATCH22_EXTERNAL_DIR`
(default `$CATCH22_ROOT/external`) and checks out a pinned commit for each. These
provide the SemStamp, PMark, and HeavyWater / SimplexWater implementations that
the pipeline adapts.

| Repository | Provides |
| --- | --- |
| `SemStamp` | Semantic `semstamp` method (ships evaluation data as a submodule) |
| `PMark` | Semantic `pmark` method |
| `HeavyWater_SimplexWater` | Distortion-free `heavywater` / `simplexwater` methods |

The `external/` directory is git-ignored, so these repositories are never
committed. Re-running is idempotent: existing clones are fetched and
re-checked-out at the pinned commit. `setup_env.sh` invokes this script; run it
standalone to refresh only the upstream repositories:

```bash
source env.sh
scripts/fetch_external.sh
```

After cloning, install each upstream repository's own Python requirements into
the same environment; the script prints the corresponding `pip install -r`
commands.

## `sync_to_cluster.sh`

Mirrors the repository to a remote cluster with `rsync -av`. Set the target in
`env.sh`:

```bash
export CATCH22_REMOTE="user@host:/path/on/cluster/"
```

then:

```bash
source env.sh
scripts/sync_to_cluster.sh
```

Run artifacts, caches, and local configuration are excluded so that only code
and manifests are transferred: `.git/`, `__pycache__/`, `*.pyc`, `external/`,
`hf_cache/`, `results/`, `outputs/`, `logs/`, and `env.sh`. Because `external/`
is excluded, run `scripts/fetch_external.sh` on the cluster after the first sync.
